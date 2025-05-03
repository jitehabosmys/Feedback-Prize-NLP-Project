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
import importlib.util
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

# 添加项目根目录到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 默认配置
from src.config.config import CFG as DefaultCFG
from src.utils.common import get_logger, seed_everything, AverageMeter, timeSince, collate, LOGGER
from src.utils.metrics import get_score, MCRMSE

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
    # 必须保留的参数
    parser.add_argument("--fold", type=int, default=0, help="使用的折数，单折训练时使用")
    parser.add_argument("--train_all_data", action="store_true", help="是否使用所有数据训练")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式")
    parser.add_argument("--model", type=str, default=None, help="模型名称")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    
    # 可选保留的参数
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument("--print_freq", type=int, default=None, help="训练过程中打印频率")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮次")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="梯度裁剪阈值")
    
    # 配置文件参数
    parser.add_argument("--config", type=str, default="default", 
                        help="配置文件路径，使用'default'表示使用默认配置")
    
    # Wandb参数
    parser.add_argument("--use_wandb", action="store_true", help="是否使用Weights & Biases记录实验")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb项目名称")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb组织名称")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb运行名称")
    parser.add_argument("--wandb_watch_model", action="store_true", help="是否使用wandb watch跟踪模型")
    
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
    
    # 导入需要的模块 - 注意这里需要动态导入，因为此时CFG可能已经被修改
    from src.data.dataset import TrainDataset, get_train_dataloader, get_valid_dataloader
    from src.data.preprocessing import prepare_folds
    from src.models.model import FeedbackModel
    from src.training.trainer import train_fn, valid_fn, train_loop
    
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
    
    # 加载配置
    global CFG
    CFG = load_config(args.config)
    
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
    
    if args.seed:
        CFG.seed = args.seed
    
    if args.output_dir:
        CFG.OUTPUT_DIR = args.output_dir
        
    if args.print_freq:
        CFG.print_freq = args.print_freq
        
    if args.epochs:
        CFG.epochs = args.epochs
    
    if args.max_grad_norm:
        CFG.max_grad_norm = args.max_grad_norm
    
    # 创建输出目录
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(CFG.OUTPUT_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(CFG.OUTPUT_DIR, 'tokenizer'), exist_ok=True)
    
    # 初始化Wandb（如果启用）
    if args.use_wandb or CFG.use_wandb:
        try:
            import wandb
            # 设置Wandb配置
            CFG.use_wandb = True
            if args.wandb_project:
                CFG.wandb_project = args.wandb_project
            if args.wandb_entity:
                CFG.wandb_entity = args.wandb_entity
            if args.wandb_run_name:
                CFG.wandb_run_name = args.wandb_run_name
            if args.wandb_watch_model:
                CFG.wandb_watch_model = True
                
            # 自动生成运行名称（如果未指定）
            run_name = CFG.wandb_run_name or f"{CFG.model_name.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}"
            
            # 如果debug模式，添加标记
            if args.debug:
                run_name = f"debug_{run_name}"
            
            # 提取配置信息为dict
            config_dict = {k: v for k, v in CFG.__dict__.items() if not k.startswith('__')}
            
            # 初始化Wandb
            LOGGER.info(f"初始化Wandb: 项目={CFG.wandb_project}, 实体={CFG.wandb_entity}, 运行名称={run_name}")
            wandb.init(
                project=CFG.wandb_project,
                entity=CFG.wandb_entity,
                name=run_name,
                config=config_dict,
            )
            
            # 记录主要配置信息
            LOGGER.info("Wandb初始化成功")
        except ImportError:
            LOGGER.warning("未安装wandb包，请先安装: pip install wandb")
            CFG.use_wandb = False
        except Exception as e:
            LOGGER.warning(f"Wandb初始化失败: {str(e)}")
            CFG.use_wandb = False
    
    # 设置日志
    LOGGER.info(f"============ 训练开始 ============")
    LOGGER.info(f"使用配置: {args.config if args.config != 'default' else '默认配置'}")
    LOGGER.info(f"模型: {CFG.model_name}")
    LOGGER.info(f"批次大小: {CFG.batch_size}")
    LOGGER.info(f"交叉验证折数: {CFG.num_folds}")
    LOGGER.info(f"最大序列长度: {CFG.max_len}")
    LOGGER.info(f"编码器学习率: {CFG.encoder_lr}")
    LOGGER.info(f"解码器学习率: {CFG.decoder_lr}")
    LOGGER.info(f"学习率调度器: {CFG.scheduler}")
    LOGGER.info(f"训练轮数: {CFG.epochs}")
    LOGGER.info(f"设备: {CFG.device}")
    LOGGER.info(f"Wandb记录: {'启用' if CFG.use_wandb else '禁用'}")
    
    # 设置种子
    seed_everything(CFG.seed)
    
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
        
        # 如果使用wandb，记录每个折的得分
        if CFG.use_wandb:
            try:
                import wandb
                wandb.log({f"fold_{fold}/score": score})
            except:
                LOGGER.warning("Wandb日志记录失败，跳过")
    
    # 保存整体OOF结果
    oof_df.to_csv(os.path.join(CFG.OUTPUT_DIR, 'oof_df.csv'), index=False)
    
    # 计算整体CV分数
    valid_labels = oof_df[CFG.target_cols].values
    valid_preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
    score, scores = get_score(valid_labels, valid_preds)
    
    LOGGER.info(f"========== 整体CV得分: {score:.4f} ==========")
    LOGGER.info(f"单项得分: {scores}")
    
    # 如果使用wandb，记录整体CV分数
    if CFG.use_wandb:
        try:
            import wandb
            # 记录最终得分
            final_metrics = {
                "final/cv_score": score,
            }
            # 记录每个目标的得分
            for i, target in enumerate(CFG.target_cols):
                final_metrics[f"final/score_{target}"] = scores[i]
                
            wandb.log(final_metrics)
            
            # 完成wandb运行
            wandb.finish()
        except:
            LOGGER.warning("Wandb日志记录失败，跳过")
    
    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main() 