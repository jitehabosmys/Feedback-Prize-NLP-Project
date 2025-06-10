import os
import gc
import time
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer

from ..config.config import CFG
from ..utils.common import AverageMeter, timeSince, collate, LOGGER
from ..utils.metrics import get_score
from ..models.model import FeedbackModel
from ..data.dataset import TrainDataset, get_train_dataloader, get_valid_dataloader

def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    """训练一个epoch"""
    model.train()
    scaler = torch.amp.GradScaler(enabled=CFG.apex)  # 根据CFG.apex决定是否启用AMP
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    
    # 如果使用wandb，尝试watch模型
    if CFG.use_wandb and CFG.wandb_watch_model and global_step == 0:
        try:
            import wandb
            wandb.watch(model, log="all")
        except:
            LOGGER.warning("尝试使用wandb.watch失败，跳过")
    
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        
        batch_size = labels.size(0)
        
        # 根据是否使用AMP决定前向传播方式
        if CFG.apex:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                y_preds = model(inputs)
                loss = criterion(y_preds, labels)
        else:
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
            
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
            
        losses.update(loss.item(), batch_size)
        
        # 根据是否使用AMP决定反向传播和梯度裁剪方式
        if CFG.apex:
            # 使用AMP时的反向传播
            scaler.scale(loss).backward()
            
            # 使用原始笔记本中的梯度裁剪方法（直接对缩放后的梯度裁剪）
            if CFG.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            else:
                grad_norm = 0.0
            
            if (step + 1) % CFG.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                if CFG.batch_scheduler:
                    scheduler.step()
        else:
            # 不使用AMP时的反向传播
            loss.backward()
            
            # 不使用AMP时的梯度裁剪
            if CFG.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            else:
                grad_norm = 0.0
            
            if (step + 1) % CFG.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if CFG.batch_scheduler:
                    scheduler.step()
                
        end = time.time()
        
        # 记录训练进度
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0],
                          ))
            
        # 如果使用wandb，记录训练指标
        if CFG.use_wandb and (step % CFG.wandb_log_interval == 0 or step == (len(train_loader)-1)):
            try:
                import wandb
                wandb.log({
                    "train/loss": losses.val,
                    "train/avg_loss": losses.avg,
                    "train/grad_norm": grad_norm,
                    "train/lr": scheduler.get_lr()[0],
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                    "train/amp_enabled": CFG.apex,
                })
            except:
                LOGGER.warning("Wandb日志记录失败，跳过")
            
    return losses.avg

def valid_fn(valid_loader, model, criterion, device):
    """验证函数"""
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    
    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        
        batch_size = labels.size(0)
        
        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
            
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
            
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())
        end = time.time()
        
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
            
        # 如果使用wandb，记录验证指标
        if CFG.use_wandb and (step % CFG.wandb_log_interval == 0 or step == (len(valid_loader)-1)):
            try:
                import wandb
                wandb.log({
                    "valid/loss": losses.val,
                    "valid/avg_loss": losses.avg,
                    "valid/step": step,
                })
            except:
                LOGGER.warning("Wandb日志记录失败，跳过")
    
    predictions = np.concatenate(preds)
    return losses.avg, predictions

# 添加自定义LogCoshLoss类
class LogCoshLoss(torch.nn.Module):
    """Log-Cosh损失函数
    
    Log-cosh是平滑的类似于均方误差的损失函数，
    对于小误差接近MSE，对于大误差接近MAE，但处处二阶可导。
    计算公式: log(cosh(x)) 其中 x = y_pred - y_true
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """计算Log-Cosh损失
        
        Args:
            y_pred: 预测值
            y_true: 真实值
            
        Returns:
            损失值
        """
        # 添加一个小的epsilon值以避免数值不稳定
        epsilon = 1e-12
        # 计算预测值和真实值之间的差异
        diff = y_pred - y_true
        # 使用log(cosh(x))公式计算损失
        loss = torch.log(torch.cosh(diff) + epsilon)
        # 返回平均损失
        return torch.mean(loss)

def train_loop(folds, fold):
    """训练循环"""
    LOGGER.info(f"========== fold: {fold} training ==========")
    
    # ====================================================
    # 加载并保存tokenizer
    # ====================================================
    LOGGER.info(f"加载并保存tokenizer: {CFG.model_name}")
    
    # 检查是否已指定tokenizer目录
    if hasattr(CFG, 'tokenizer_dir') and CFG.tokenizer_dir and os.path.exists(CFG.tokenizer_dir):
        LOGGER.info(f"使用指定目录的tokenizer: {CFG.tokenizer_dir}")
        tokenizer = AutoTokenizer.from_pretrained(CFG.tokenizer_dir, local_files_only=True)
    else:
        # 创建tokenizer目录
        tokenizer_dir = os.path.join(CFG.MODEL_OUTPUT_DIR, 'tokenizer')
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        # 检查是否为离线模式
        local_files_only = getattr(CFG, 'local_files_only', False)
        
        try:
            # 尝试加载并保存tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                CFG.model_name, 
                local_files_only=local_files_only
            )
            tokenizer.save_pretrained(tokenizer_dir)
            LOGGER.info(f"Tokenizer已保存到: {tokenizer_dir}")
            # 设置tokenizer目录配置
            CFG.tokenizer_dir = tokenizer_dir
        except Exception as e:
            # 如果加载失败，尝试查找已有的tokenizer
            LOGGER.warning(f"无法从网络加载tokenizer: {str(e)}")
            model_name_safe = CFG.model_name.replace('/', '-')
            
            # 尝试在不同位置查找tokenizer
            possible_dirs = [
                os.path.join(CFG.OUTPUT_DIR, model_name_safe, 'tokenizer'),  # 当前输出目录下的模型子目录
                os.path.join(CFG.OUTPUT_DIR, 'tokenizer'),                  # 旧版目录结构
                os.path.join(os.path.dirname(CFG.OUTPUT_DIR), model_name_safe, 'tokenizer')  # 上级目录
            ]
            
            tokenizer = None
            for dir_path in possible_dirs:
                if os.path.exists(dir_path):
                    try:
                        LOGGER.info(f"尝试从本地目录加载tokenizer: {dir_path}")
                        tokenizer = AutoTokenizer.from_pretrained(dir_path, local_files_only=True)
                        CFG.tokenizer_dir = dir_path
                        LOGGER.info(f"成功从本地目录加载tokenizer: {dir_path}")
                        break
                    except Exception as e2:
                        LOGGER.warning(f"从目录 {dir_path} 加载tokenizer失败: {str(e2)}")
            
            if tokenizer is None:
                raise ValueError(f"无法加载tokenizer，请确保网络连接或提供有效的tokenizer目录。原始错误: {str(e)}")
    
    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)
    
    train_loader = get_train_dataloader(train_dataset, CFG.batch_size, CFG.num_workers)
    valid_loader = get_valid_dataloader(valid_dataset, CFG.batch_size, CFG.num_workers)
    
    # ====================================================
    # model & optimizer
    # ====================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 检查是否为离线模式
    local_files_only = getattr(CFG, 'local_files_only', False)
    
    # 获取配置文件路径
    config_path = getattr(CFG, 'config_path', None)
    
    # 创建模型
    model = FeedbackModel(
        CFG.model_name, 
        pooling_type=CFG.pooling_type,
        local_files_only=local_files_only,
        config_path=config_path
    )
    
    # 修改保存配置的部分
    try:
        # 保存配置文件
        if hasattr(model, 'config'):
            LOGGER.info(f"保存模型配置到: {os.path.join(CFG.MODEL_OUTPUT_DIR, 'config.pth')}")
            torch.save(model.config, os.path.join(CFG.MODEL_OUTPUT_DIR, 'config.pth'))
        elif hasattr(model.backbone, 'config'):
            LOGGER.info(f"保存backbone配置到: {os.path.join(CFG.MODEL_OUTPUT_DIR, 'config.pth')}")
            torch.save(model.backbone.config, os.path.join(CFG.MODEL_OUTPUT_DIR, 'config.pth'))
        else:
            LOGGER.warning("无法找到模型配置，跳过保存config.pth")
        
        # 检查预训练模型目录是否已存在
        pretrained_dir = os.path.join(CFG.MODEL_OUTPUT_DIR, 'pretrained_model')
        if not os.path.exists(pretrained_dir) and not getattr(CFG, 'local_files_only', False):
            # 只在有网络环境且预训练模型目录不存在时保存预训练模型
            if hasattr(model, 'backbone'):
                LOGGER.info(f"保存官方预训练模型到: {pretrained_dir}")
                os.makedirs(pretrained_dir, exist_ok=True)
                model.backbone.save_pretrained(pretrained_dir)
            else:
                LOGGER.warning("无法找到backbone，跳过保存预训练模型")
        elif os.path.exists(pretrained_dir):
            LOGGER.info(f"预训练模型目录已存在: {pretrained_dir}，跳过保存")
        else:
            LOGGER.info("离线模式，跳过保存预训练模型")
            
    except Exception as e:
        LOGGER.warning(f"保存配置或预训练模型失败: {str(e)}")
    
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "backbone" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters
    
    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    
    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)
    
    # ====================================================
    # loop
    # ====================================================
    # 根据配置选择损失函数
    if hasattr(CFG, 'loss_type'):
        if CFG.loss_type == 'mse':
            LOGGER.info("使用MSE损失函数")
            criterion = torch.nn.MSELoss(reduction='mean')
        elif CFG.loss_type == 'log_cosh':
            LOGGER.info("使用Log-Cosh损失函数")
            criterion = LogCoshLoss()
        else:
            LOGGER.info("使用SmoothL1Loss损失函数")
            criterion = torch.nn.SmoothL1Loss(reduction='mean')
    else:
        LOGGER.info("使用默认SmoothL1Loss损失函数")
        criterion = torch.nn.SmoothL1Loss(reduction='mean')  # RMSELoss(reduction="mean")
    
    best_score = np.inf
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        
        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)
        
        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        
        # scoring
        score, scores = get_score(valid_labels, predictions)
        
        elapsed = time.time() - start_time
        
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
        
        # 如果使用wandb，记录每个epoch的结果
        if CFG.use_wandb:
            try:
                import wandb
                # 记录epoch级别指标
                epoch_log = {
                    "epoch": epoch + 1,
                    "train/epoch_loss": avg_loss,
                    "valid/epoch_loss": avg_val_loss,
                    "valid/score": score,
                    "time_per_epoch": elapsed,
                }
                # 记录每个目标的得分
                for i, target in enumerate(CFG.target_cols):
                    epoch_log[f"valid/score_{target}"] = scores[i]
                
                wandb.log(epoch_log)
            except:
                LOGGER.warning("Wandb日志记录失败，跳过")
        
        if best_score > score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        os.path.join(CFG.MODEL_OUTPUT_DIR, f"models/{CFG.model_name.replace('/', '-')}_fold{fold}_best.pth"))
            
            # 如果使用wandb，记录最佳模型信息
            if CFG.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "valid/best_score": best_score,
                        "valid/best_epoch": epoch + 1,
                    })
                    # 可选：上传最佳模型文件到wandb
                    # model_path = os.path.join(CFG.OUTPUT_DIR, f"models/{CFG.model_name.replace('/', '-')}_fold{fold}_best.pth")
                    # wandb.save(model_path, base_path=CFG.OUTPUT_DIR)
                except:
                    LOGGER.warning("Wandb日志记录失败，跳过")
    
    predictions = torch.load(os.path.join(CFG.MODEL_OUTPUT_DIR, f"models/{CFG.model_name.replace('/', '-')}_fold{fold}_best.pth"),
                          map_location=torch.device('cpu'), weights_only=False)['predictions']
    
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds 