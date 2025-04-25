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
import importlib.util
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig

# 添加项目根目录到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 默认配置
from src.config.config import CFG as DefaultCFG
from src.utils.common import seed_everything, LOGGER

# 配置对象，将在main中根据命令行参数设置
CFG = None

def load_config(config_path):
    """根据路径动态加载配置文件"""
    try:
        if config_path == "default":
            return DefaultCFG
            
        # 使用importlib动态加载指定配置文件
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # 检查加载的模块是否包含CFG对象
        if hasattr(config_module, "CFG"):
            LOGGER.info(f"成功加载配置文件: {config_path}")
            return config_module.CFG
        else:
            LOGGER.warning(f"配置文件 {config_path} 中未找到CFG对象，使用默认配置")
            return DefaultCFG
    except Exception as e:
        LOGGER.error(f"加载配置文件 {config_path} 失败: {str(e)}")
        LOGGER.info("使用默认配置")
        return DefaultCFG

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser()
    # 必要参数
    parser.add_argument("--model", type=str, default=None, help="模型名称")
    parser.add_argument("--model_dir", type=str, default=None, help="模型文件目录")
    parser.add_argument("--output_file", type=str, default="submission.csv", help="输出文件名")
    parser.add_argument("--num_folds", type=int, default=None, help="使用多少折模型进行集成")
    
    # 可选参数
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    
    # 配置文件参数
    parser.add_argument("--config", type=str, default="default", 
                        help="配置文件路径，使用'default'表示使用默认配置")
    
    return parser.parse_args()

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
    
    # 加载配置
    global CFG
    CFG = load_config(args.config)
    
    # 设置配置
    if args.model:
        CFG.model_name = args.model
    
    if args.batch_size:
        CFG.batch_size = args.batch_size
    
    if args.seed:
        CFG.seed = args.seed
    
    if args.model_dir:
        CFG.OUTPUT_DIR = args.model_dir
        
    if args.num_folds:
        CFG.num_folds = args.num_folds
    
    # 创建输出目录
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(CFG.OUTPUT_DIR, 'results'), exist_ok=True)
    
    # 准备输出文件路径
    output_file_path = os.path.join(CFG.OUTPUT_DIR, 'results', args.output_file)
    
    # 设置日志
    LOGGER.info(f"============ 预测开始 ============")
    LOGGER.info(f"使用配置: {args.config if args.config != 'default' else '默认配置'}")
    LOGGER.info(f"模型: {CFG.model_name}")
    LOGGER.info(f"批次大小: {CFG.batch_size}")
    LOGGER.info(f"最大序列长度: {CFG.max_len}")
    LOGGER.info(f"交叉验证折数: {CFG.num_folds}")
    LOGGER.info(f"输出文件: {output_file_path}")
    
    # 设置种子
    seed_everything(CFG.seed)
    
    # 导入依赖模块 - 在CFG设置完成后导入
    from src.data.dataset import TestDataset, get_test_dataloader, get_tokenizer
    from src.models.model import FeedbackModel
    
    # 加载测试数据
    test_df = pd.read_csv(os.path.join(CFG.DATA_DIR, 'test.csv'))
    submission = pd.read_csv(os.path.join(CFG.DATA_DIR, 'sample_submission.csv'))
    
    # 准备数据集和加载器
    test_dataset = TestDataset(CFG, test_df)
    tokenizer = get_tokenizer(CFG.model_name)
    test_loader = get_test_dataloader(test_dataset, CFG.batch_size, CFG.num_workers, tokenizer)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model_paths = []
    
    # 获取所有模型文件
    for fold in range(CFG.num_folds):
        model_path = os.path.join(
            CFG.OUTPUT_DIR, 
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
    submission.to_csv(output_file_path, index=False)
    LOGGER.info(f"预测结果已保存到 {output_file_path}")

if __name__ == "__main__":
    main() 