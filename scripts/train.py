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
from src.utils.common import get_logger, seed_everything, AverageMeter, timeSince, collate, LOGGER
from src.utils.metrics import get_score, MCRMSE
from src.models.model import FeedbackModel
from src.training.trainer import train_fn, valid_fn, train_loop

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=CFG.seed, help="随机种子")
    parser.add_argument("--fold", type=int, default=0, help="使用的折数，单折训练时使用")
    parser.add_argument("--model", type=str, default=CFG.model_name, help="模型名称")
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size, help="批次大小")
    parser.add_argument("--epochs", type=int, default=CFG.epochs, help="训练轮数")
    parser.add_argument("--lr", type=float, default=CFG.learning_rate, help="学习率")
    parser.add_argument("--output_dir", type=str, default=CFG.OUTPUT_DIR, help="输出目录")
    parser.add_argument("--train_all_data", action="store_true", help="是否使用所有数据训练")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式")
    return parser.parse_args()

def train_single_fold(args, fold):
    """训练单折"""
    LOGGER.info(f"========== 第 {fold} 折训练 ==========")
    
    # 加载数据
    train_df = pd.read_csv(os.path.join(CFG.DATA_DIR, 'train.csv'))
    
    if args.debug:
        train_df = train_df.sample(n=100, random_state=CFG.seed).reset_index(drop=True)
    
    # 准备交叉验证
    Fold = MultilabelStratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(X=train_df, y=train_df[CFG.target_cols])):
        train_df.loc[val_index, 'fold'] = int(n)
    train_df['fold'] = train_df['fold'].astype(int)
    
    # 运行训练
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
    
    if args.model:
        CFG.model_name = args.model
    
    if args.batch_size:
        CFG.batch_size = args.batch_size
    
    if args.lr:
        CFG.learning_rate = args.lr
    
    if args.output_dir:
        CFG.OUTPUT_DIR = args.output_dir
    
    # 创建输出目录
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(CFG.OUTPUT_DIR, 'models'), exist_ok=True)
    
    # 设置日志
    LOGGER.info(f"============ 训练开始 ============")
    LOGGER.info(f"模型: {CFG.model_name}")
    LOGGER.info(f"批次大小: {CFG.batch_size}")
    LOGGER.info(f"学习率: {CFG.learning_rate}")
    LOGGER.info(f"训练轮数: {CFG.epochs}")
    
    # 设置种子
    seed_everything(args.seed)
    
    # 训练
    if args.train_all_data:
        # 使用所有数据训练
        folds = [0, 1, 2, 3, 4]  # 假设有5折
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