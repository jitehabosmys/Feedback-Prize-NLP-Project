#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练脚本 - 基于fb3-deberta-v3-base-baseline-train.ipynb
"""

import os
import gc
import re
import sys
import json
import time
import math
import random
import warnings
import argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except ImportError:
    print("正在安装MultilabelStratifiedKFold...")
    os.system('pip install iterative-stratification==0.1.7')
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config.config import CFG
from src.data.dataset import TrainDataset, get_train_dataloader, get_valid_dataloader
from src.data.preprocessing import prepare_folds
from src.utils.common import get_logger, seed_everything, AverageMeter, timeSince, collate, LOGGER
from src.utils.metrics import get_score, MCRMSE
from src.models.model import FeedbackModel
from src.training.trainer import train_fn, valid_fn, train_loop

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser()
    # 基本参数
    parser.add_argument("--seed", type=int, default=CFG.seed, help="随机种子")
    parser.add_argument("--fold", type=int, default=0, help="使用的折数，单折训练时使用")
    parser.add_argument("--num_folds", type=int, default=CFG.num_folds, help="交叉验证折数")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式")
    parser.add_argument("--train_all_data", action="store_true", help="是否使用所有数据训练")
    
    # 模型参数
    parser.add_argument("--model", type=str, default=CFG.model_name, help="模型名称")
    parser.add_argument("--max_len", type=int, default=CFG.max_len, help="最大序列长度")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size, help="批次大小")
    parser.add_argument("--epochs", type=int, default=CFG.epochs, help="训练轮数")
    parser.add_argument("--encoder_lr", type=float, default=CFG.encoder_lr, help="编码器学习率")
    parser.add_argument("--decoder_lr", type=float, default=CFG.decoder_lr, help="解码器学习率")
    parser.add_argument("--weight_decay", type=float, default=CFG.weight_decay, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=CFG.num_workers, help="数据加载线程数")
    
    # 学习率调度器参数
    parser.add_argument("--scheduler", type=str, default=CFG.scheduler, choices=['linear', 'cosine'], help="学习率调度器类型")
    parser.add_argument("--warmup_steps", type=int, default=CFG.num_warmup_steps, help="预热步数")
    parser.add_argument("--num_cycles", type=float, default=CFG.num_cycles, help="cosine调度器的周期数")
    
    # 路径参数
    parser.add_argument("--output_dir", type=str, default=CFG.OUTPUT_DIR, help="输出目录")
    
    return parser.parse_args()

def train_single_fold(args, fold):
    """训练单折"""
    LOGGER.info(f"========== 第 {fold} 折训练 ==========")
    
    # 加载数据
    train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, 'train.csv'))
    
    if args.debug:
        # 在debug模式下使用更少的数据
        LOGGER.info("DEBUG模式：使用100条数据进行训练")
        train_df = train_df.sample(n=100, random_state=CFG.seed).reset_index(drop=True)
    
    # 准备交叉验证
    train_df = prepare_folds(train_df, n_fold=CFG.num_folds)
    
    # 运行训练
    LOGGER.info(f"训练数据量: {len(train_df)}")
    LOGGER.info(f"训练样例: {train_df['full_text'].values[0][:100]}...")
    valid_folds = train_loop(train_df, fold)
    
    # 返回结果
    valid_labels = valid_folds[CFG.target_cols].values
    valid_preds = valid_folds[[f"pred_{c}" for c in CFG.target_cols]].values
    score, scores = get_score(valid_labels, valid_preds)
    
    LOGGER.info(f"========== 第 {fold} 折训练结束 ==========")
    LOGGER.info(f"CV Score: {score:.4f}")
    LOGGER.info(f"CV Scores: {scores}")
    
    return valid_folds, score

def main():
    """主函数"""
    args = parse_args()
    
    # 设置配置
    if args.debug:
        CFG.epochs = 1
        CFG.debug = True
        # 在debug模式下减少打印频率
        CFG.print_freq = 10
        
    if args.model:
        CFG.model_name = args.model
    
    if args.batch_size:
        CFG.batch_size = args.batch_size
    
    if args.encoder_lr:
        CFG.encoder_lr = args.encoder_lr
        
    if args.decoder_lr:
        CFG.decoder_lr = args.decoder_lr
        
    if args.weight_decay:
        CFG.weight_decay = args.weight_decay
        
    if args.max_len:
        CFG.max_len = args.max_len
        
    if args.num_folds:
        CFG.num_folds = args.num_folds
        
    if args.scheduler:
        CFG.scheduler = args.scheduler
        
    if args.warmup_steps:
        CFG.num_warmup_steps = args.warmup_steps
        
    if args.num_cycles:
        CFG.num_cycles = args.num_cycles
        
    if args.num_workers:
        CFG.num_workers = args.num_workers
    
    if args.output_dir:
        CFG.OUTPUT_DIR = args.output_dir
    
    # 创建输出目录
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(CFG.OUTPUT_DIR, 'models'), exist_ok=True)
    
    # 设置日志
    LOGGER.info(f"============ 训练开始 ============")
    LOGGER.info(f"模型: {CFG.model_name}")
    LOGGER.info(f"批次大小: {CFG.batch_size}")
    LOGGER.info(f"编码器学习率: {CFG.encoder_lr}")
    LOGGER.info(f"解码器学习率: {CFG.decoder_lr}")
    LOGGER.info(f"交叉验证折数: {CFG.num_folds}")
    LOGGER.info(f"最大序列长度: {CFG.max_len}")
    LOGGER.info(f"学习率调度器: {CFG.scheduler}")
    LOGGER.info(f"训练轮数: {CFG.epochs}")
    LOGGER.info(f"设备: {CFG.device}")
    
    # 设置种子
    seed_everything(args.seed)
    
    # 预加载常用组件
    from src.data.dataset import get_tokenizer
    LOGGER.info("预加载tokenizer...")
    get_tokenizer(CFG.model_name)
    
    # 训练
    if args.train_all_data:
        # 使用所有数据训练
        folds = list(range(CFG.num_folds))
    else:
        # 只使用特定折
        folds = [args.fold]
    
    # 初始化保存所有验证集预测结果的DataFrame
    oof_df = pd.DataFrame()
    
    for fold in folds:
        _oof_df, score = train_single_fold(args, fold)
        oof_df = pd.concat([oof_df, _oof_df])
        LOGGER.info(f"========== 第 {fold} 折得分: {score} ==========")
    
    # 保存整体OOF结果
    oof_df.to_csv(os.path.join(CFG.OUTPUT_DIR, 'oof_df.csv'), index=False)
    
    # 计算整体CV分数
    valid_labels = oof_df[CFG.target_cols].values
    valid_preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
    score, scores = get_score(valid_labels, valid_preds)
    
    LOGGER.info(f"========== 整体CV得分: {score:.4f} ==========")
    LOGGER.info(f"单项得分: {scores}")
    
    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main() 