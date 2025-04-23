import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from ..config.config import CFG
from ..data.dataset import prepare_loaders
from ..models.model import FeedbackModel


def train_one_epoch(model, optimizer, scheduler, criterion, loader):
    """
    训练一个epoch
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        criterion: 损失函数
        loader: 数据加载器
        
    Returns:
        平均损失
    """
    model.train()
    losses = []
    
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(CFG.device)
        attention_mask = batch['attention_mask'].to(CFG.device)
        token_type_ids = batch['token_type_ids'].to(CFG.device)
        targets = batch['targets'].to(CFG.device)
        
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, targets)
        
        loss.backward()
        
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
    
    return np.mean(losses)


def valid_one_epoch(model, criterion, loader):
    """
    验证一个epoch
    
    Args:
        model: 模型
        criterion: 损失函数
        loader: 数据加载器
        
    Returns:
        平均损失和RMSE分数
    """
    model.eval()
    losses = []
    preds = []
    targets_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            input_ids = batch['input_ids'].to(CFG.device)
            attention_mask = batch['attention_mask'].to(CFG.device)
            token_type_ids = batch['token_type_ids'].to(CFG.device)
            targets = batch['targets'].to(CFG.device)
            
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, targets)
            
            losses.append(loss.item())
            preds.append(outputs.detach().cpu().numpy())
            targets_list.append(targets.detach().cpu().numpy())
    
    preds = np.concatenate(preds)
    targets_list = np.concatenate(targets_list)
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(targets_list, preds))
    
    return np.mean(losses), rmse


def train_loop(df, fold, tokenizer_path):
    """
    训练循环
    
    Args:
        df: 数据DataFrame
        fold: 当前折数
        tokenizer_path: 分词器路径
        
    Returns:
        None
    """
    print(f"Training Fold {fold}")
    
    # 准备数据加载器
    train_loader, valid_loader = prepare_loaders(
        df, 
        fold, 
        tokenizer_path, 
        CFG.max_len, 
        CFG.target_cols, 
        CFG.batch_size, 
        CFG.num_workers
    )
    
    # 初始化模型
    model = FeedbackModel(CFG.model_name, len(CFG.target_cols))
    model.to(CFG.device)
    
    # 定义优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay
    )
    
    # 计算总步数
    num_train_steps = int(len(train_loader) * CFG.epochs)
    
    # 初始化学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_train_steps,
        eta_min=CFG.min_lr
    )
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 保存验证损失最低的模型
    best_val_loss = float('inf')
    best_rmse = float('inf')
    
    # 开始训练
    for epoch in range(CFG.epochs):
        print(f"Epoch {epoch + 1}/{CFG.epochs}")
        
        # 训练一个epoch
        train_loss = train_one_epoch(model, optimizer, scheduler, criterion, train_loader)
        
        # 验证一个epoch
        val_loss, rmse = valid_one_epoch(model, criterion, valid_loader)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, RMSE: {rmse:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"model_fold{fold}_best_loss.pth")
            print(f"Saved model with best val loss: {best_val_loss:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), f"model_fold{fold}_best_rmse.pth")
            print(f"Saved model with best RMSE: {best_rmse:.4f}")
    
    return best_val_loss, best_rmse 