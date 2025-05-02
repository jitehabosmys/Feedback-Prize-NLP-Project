import os
import gc
import time
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from ..config.config import CFG
from ..utils.common import AverageMeter, timeSince, collate, LOGGER
from ..utils.metrics import get_score
from ..models.model import FeedbackModel
from ..data.dataset import TrainDataset, get_train_dataloader, get_valid_dataloader

def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    """训练一个epoch"""
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=CFG.apex)  # 使用GPU进行混合精度训练...神奇的是，即使pytorch版本是cpu，也能正常使用
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
        
        with torch.amp.autocast('cuda', enabled=CFG.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
            
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
            
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        
        # 使用原始笔记本中的梯度裁剪方法（直接对缩放后的梯度裁剪）
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        
        # 正确的实现方式（现在注释掉）
        # scaler.unscale_(optimizer)
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
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
                          lr=scheduler.get_lr()[0]))
            
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

def train_loop(folds, fold):
    """训练循环"""
    LOGGER.info(f"========== fold: {fold} training ==========")
    
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
    model = FeedbackModel(CFG.model_name)
    torch.save(model.backbone.config, os.path.join(CFG.OUTPUT_DIR, 'config.pth'))
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
                        os.path.join(CFG.OUTPUT_DIR, f"models/{CFG.model_name.replace('/', '-')}_fold{fold}_best.pth"))
            
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
    
    predictions = torch.load(os.path.join(CFG.OUTPUT_DIR, f"models/{CFG.model_name.replace('/', '-')}_fold{fold}_best.pth"),
                          map_location=torch.device('cpu'), weights_only=False)['predictions']
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds 