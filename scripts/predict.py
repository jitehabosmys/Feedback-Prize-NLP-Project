#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预测脚本 - 基于fb3-deberta-v3-base-baseline-inference.ipynb
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config.config import CFG
from src.data.dataset import TestDataset, get_test_dataloader
from src.utils.common import seed_everything, LOGGER
from src.models.model import FeedbackModel

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser()
    # 基本参数
    parser.add_argument("--seed", type=int, default=CFG.seed, help="随机种子")
    parser.add_argument("--num_folds", type=int, default=CFG.num_folds, help="交叉验证折数")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式")
    
    # 模型参数
    parser.add_argument("--model", type=str, default=CFG.model_name, help="模型名称")
    parser.add_argument("--max_len", type=int, default=CFG.max_len, help="最大序列长度")
    
    # 预测参数
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=CFG.num_workers, help="数据加载线程数")
    parser.add_argument("--use_tta", action="store_true", help="是否使用测试时增强")
    
    # 路径参数
    parser.add_argument("--model_dir", type=str, default=CFG.OUTPUT_DIR, help="模型文件目录")
    parser.add_argument("--output_file", type=str, default="submission.csv", help="输出文件名")
    
    return parser.parse_args()

class TestDataset(Dataset):
    """测试数据集"""
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.cfg.max_len,
            padding='max_length',
            return_tensors=None,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
        }

def inference_fn(test_loader, model, device):
    """推理函数"""
    preds = []
    model.eval()
    model.to(device)
    
    tk0 = tqdm(test_loader, desc="推理中")
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    
    predictions = np.concatenate(preds)
    return predictions

def main():
    """主函数"""
    args = parse_args()
    
    # 设置配置
    if args.debug:
        CFG.debug = True
    
    if args.model:
        CFG.model_name = args.model
    
    if args.batch_size:
        CFG.batch_size = args.batch_size
    
    if args.max_len:
        CFG.max_len = args.max_len
    
    if args.model_dir:
        CFG.model_dir = args.model_dir
        
    if args.num_folds:
        CFG.num_folds = args.num_folds
        
    if args.num_workers:
        CFG.num_workers = args.num_workers
    
    # 设置日志
    LOGGER.info(f"============ 预测开始 ============")
    LOGGER.info(f"模型: {CFG.model_name}")
    LOGGER.info(f"批次大小: {CFG.batch_size}")
    LOGGER.info(f"最大序列长度: {CFG.max_len}")
    LOGGER.info(f"交叉验证折数: {CFG.num_folds}")
    LOGGER.info(f"是否使用TTA: {args.use_tta}")
    
    # 设置种子
    seed_everything(args.seed)
    
    # 加载测试数据
    test_df = pd.read_csv(os.path.join(CFG.DATA_DIR, 'test.csv'))
    submission = pd.read_csv(os.path.join(CFG.DATA_DIR, 'sample_submission.csv'))
    
    if args.debug:
        test_df = test_df.head(100)
    
    # 准备数据集和加载器
    test_dataset = TestDataset(CFG, test_df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model_paths = []
    
    # 获取所有模型文件
    for fold in range(CFG.num_folds):
        model_path = os.path.join(
            CFG.model_dir, 
            f"models/{CFG.model_name.replace('/', '-')}_fold{fold}_best.pth"
        )
        if os.path.exists(model_path):
            model_paths.append(model_path)
            LOGGER.info(f"找到模型: {model_path}")
    
    if not model_paths:
        LOGGER.error("未找到任何模型文件！")
        sys.exit(1)
    
    # 运行推理
    final_preds = []
    
    for i, model_path in enumerate(model_paths):
        LOGGER.info(f"使用模型 {i+1}/{len(model_paths)}: {model_path}")
        
        # 初始化模型
        model = FeedbackModel(CFG.model_name)
        
        # 加载模型权重
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        if 'model' in state:
            model.load_state_dict(state['model'])
            LOGGER.info(f"加载模型权重成功")
        else:
            model.load_state_dict(state)
            LOGGER.info(f"加载模型权重成功")
        
        # 运行推理
        predictions = inference_fn(test_loader, model, device)
        final_preds.append(predictions)
        
        # 清理内存
        torch.cuda.empty_cache()
        del model
        gc.collect()
    
    # 平均所有模型的预测结果
    final_preds = np.mean(final_preds, axis=0)
    
    # 保存预测结果
    submission[CFG.target_cols] = final_preds
    submission.to_csv(args.output_file, index=False)
    LOGGER.info(f"预测结果已保存到 {args.output_file}")

if __name__ == "__main__":
    main() 