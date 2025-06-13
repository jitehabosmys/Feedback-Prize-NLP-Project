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

from sklearn.metrics import mean_squared_error
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
    parser.add_argument("--output_dir", type=str, default="./results", help="输出结果保存目录")
    parser.add_argument("--output_file", type=str, default="submission.csv", help="输出文件名")
    parser.add_argument("--num_folds", type=int, default=None, help="使用多少折模型进行集成")
    
    # 可选参数
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录")
    parser.add_argument("--tokenizer_dir", type=str, default=None, 
                        help="tokenizer目录，如果指定，将优先使用此目录中的tokenizer")
    parser.add_argument("--config_path", type=str, default=None,
                        help="模型配置文件路径，优先使用此配置；离线模式下必须提供或自动找到")
    parser.add_argument("--local_files_only", action="store_true", 
                        help="仅使用本地文件，不下载（在离线环境如Kaggle推理中使用）")
    
    # 池化和初始化参数
    parser.add_argument("--pooling_type", type=str, default=None, 
                        choices=['mean', 'cls', 'attention', 'weighted_layer'],
                        help="池化方式: mean, cls, attention, weighted_layer")
    
    # 配置文件参数
    parser.add_argument("--config", type=str, default="default", 
                        help="配置文件路径，使用'default'表示使用默认配置")
    
    # Stacking
    parser.add_argument("--stacking", action="store_true", help="是否使用stacking进行集成")
    parser.add_argument("--stacking_dir", type=str, default="./stacking", help="stacking数据目录，存放oof和submission文件")
    parser.add_argument("--stacking_method", type=str, default="bayesian", choices=["bayesian", "multitask"], 
                    help="stacking方法: bayesian (每个任务单独使用贝叶斯回归), multitask (使用MultiTaskLasso)")
    parser.add_argument("--stacking_cv", type=int, default=5, help="stacking中使用的交叉验证折数")
    parser.add_argument("--stacking_repeats", type=int, default=2, help="stacking中交叉验证的重复次数")
    
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

def load_stacking_data(stacking_dir):
    """加载stacking所需的数据"""
    oof_files = sorted([f for f in os.listdir(stacking_dir) if f.startswith('oof_df_') and f.endswith('.csv')])
    submission_files = sorted([f for f in os.listdir(stacking_dir) if f.startswith('submission_') and f.endswith('.csv')])
    
    if len(oof_files) != len(submission_files):
        LOGGER.warning(f"oof文件数量({len(oof_files)})与submission文件数量({len(submission_files)})不一致")
    
    oof_dfs = []
    submission_dfs = []
    
    for oof_file in oof_files:
        try:
            oof_df = pd.read_csv(os.path.join(stacking_dir, oof_file))
            oof_dfs.append(oof_df)
            LOGGER.info(f"加载oof文件: {oof_file}")
        except Exception as e:
            LOGGER.error(f"加载oof文件{oof_file}失败: {str(e)}")
    
    for sub_file in submission_files:
        try:
            sub_df = pd.read_csv(os.path.join(stacking_dir, sub_file))
            submission_dfs.append(sub_df)
            LOGGER.info(f"加载submission文件: {sub_file}")
        except Exception as e:
            LOGGER.error(f"加载submission文件{sub_file}失败: {str(e)}")
    
    return oof_dfs, submission_dfs

def bayesian_stacking(oof_dfs, submission_dfs, cv_splits=5, cv_repeats=2):
    """使用贝叶斯回归进行stacking集成"""
    from sklearn.linear_model import BayesianRidge
    from sklearn.model_selection import RepeatedKFold
    
    # 准备数据
    target_cols = CFG.target_cols
    pred_cols = [f"pred_{col}" for col in target_cols]
    
    # 提取真实标签和预测值
    y_true = oof_dfs[0][target_cols].values
    
    # 构建训练特征矩阵
    X_train = np.hstack([df[pred_cols].values for df in oof_dfs])
    
    # 构建测试特征矩阵
    X_test = np.hstack([df[target_cols].values for df in submission_dfs])
    
    # 初始化预测结果
    predictions = np.zeros((X_test.shape[0], len(target_cols)))
    
    # 创建交叉验证对象
    folds = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=CFG.seed)
    
    # 用于存储交叉验证预测结果
    oof_predictions = np.zeros_like(y_true)
    fold_scores = []
    
    # 对每个目标分别进行stacking
    for i, col in enumerate(target_cols):
        LOGGER.info(f"为目标 {col} 训练stacking模型")
        
        # 提取当前目标的真实值
        y = y_true[:, i]
        
        # 提取当前目标的所有模型预测值
        X_i = np.hstack([X_train[:, i:X_train.shape[1]:len(target_cols)] for _ in range(len(oof_dfs))])
        X_test_i = np.hstack([X_test[:, i:X_test.shape[1]:len(target_cols)] for _ in range(len(submission_dfs))])
        
        # 当前目标的折得分
        col_fold_scores = []
        
        # 对当前目标进行交叉验证stacking
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_i, y)):
            LOGGER.info(f"  Fold {fold_+1}/{cv_splits*cv_repeats}")
            
            # 划分训练集和验证集
            X_train_fold, y_train_fold = X_i[trn_idx], y[trn_idx]
            X_val_fold, y_val_fold = X_i[val_idx], y[val_idx]
            
            # 训练贝叶斯回归模型
            model = BayesianRidge()
            model.fit(X_train_fold, y_train_fold)
            
            # 预测验证集并保存结果
            oof_predictions[val_idx, i] = model.predict(X_val_fold)
            
            # 预测并累加结果
            predictions[:, i] += model.predict(X_test_i) / (cv_splits * cv_repeats)
            
            # 计算当前折的当前目标的RMSE
            fold_rmse = np.sqrt(mean_squared_error(y_val_fold, oof_predictions[val_idx, i]))
            col_fold_scores.append(fold_rmse)
            LOGGER.info(f"  目标 {col} Fold {fold_+1} RMSE: {fold_rmse:.4f}")
        
        # 计算当前目标的平均折得分
        avg_col_rmse = np.mean(col_fold_scores)
        LOGGER.info(f"  目标 {col} 平均RMSE: {avg_col_rmse:.4f}")
        fold_scores.append(col_fold_scores)
    
    # 计算每个目标的RMSE
    target_scores = []
    for i, col in enumerate(target_cols):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], oof_predictions[:, i]))
        target_scores.append(rmse)
        LOGGER.info(f"目标 {col} RMSE: {rmse:.4f}")
    
    # 计算总体MCRMSE
    mcrmse = np.mean(target_scores)
    LOGGER.info(f"总体 MCRMSE: {mcrmse:.4f}")
    LOGGER.info(f"各目标RMSE: {target_scores}")
    
    return predictions, mcrmse, target_scores, oof_predictions

def multitask_stacking(oof_dfs, submission_dfs, cv_splits=5, cv_repeats=2):
    """使用MultiTaskLasso进行stacking集成"""
    from sklearn.linear_model import MultiTaskLasso
    from sklearn.model_selection import RepeatedKFold
    
    # 准备数据
    target_cols = CFG.target_cols
    pred_cols = [f"pred_{col}" for col in target_cols]
    
    # 提取真实标签和预测值
    y_true = oof_dfs[0][target_cols].values
    
    # 构建训练特征矩阵
    X_train = np.hstack([df[pred_cols].values for df in oof_dfs])
    
    # 构建测试特征矩阵
    X_test = np.hstack([df[target_cols].values for df in submission_dfs])
    
    # 初始化预测结果
    predictions = np.zeros((X_test.shape[0], len(target_cols)))
    
    # 创建交叉验证对象
    folds = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=CFG.seed)
    
    # 用于存储交叉验证预测结果
    oof_predictions = np.zeros_like(y_true)
    fold_scores = []
    
    # 进行交叉验证stacking
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_true)):
        LOGGER.info(f"Fold {fold_+1}/{cv_splits*cv_repeats}")
        
        # 划分训练集和验证集
        X_train_fold, y_train_fold = X_train[trn_idx], y_true[trn_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_true[val_idx]
        
        # 训练MultiTaskLasso模型
        model = MultiTaskLasso(alpha=0.01)
        model.fit(X_train_fold, y_train_fold)
        
        # 预测验证集并保存结果
        oof_predictions[val_idx] = model.predict(X_val_fold)
        
        # 计算当前折的MCRMSE
        fold_target_scores = []
        for i, col in enumerate(target_cols):
            fold_rmse = np.sqrt(mean_squared_error(y_val_fold[:, i], oof_predictions[val_idx, i]))
            fold_target_scores.append(fold_rmse)
        
        fold_mcrmse = np.mean(fold_target_scores)
        fold_scores.append(fold_mcrmse)
        LOGGER.info(f"Fold {fold_+1} MCRMSE: {fold_mcrmse:.4f}")
        LOGGER.info(f"Fold {fold_+1} 各目标RMSE: {fold_target_scores}")
        
        # 预测并累加结果
        predictions += model.predict(X_test) / (cv_splits * cv_repeats)
    
    # 计算每个目标的RMSE
    target_scores = []
    for i, col in enumerate(target_cols):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], oof_predictions[:, i]))
        target_scores.append(rmse)
        LOGGER.info(f"目标 {col} RMSE: {rmse:.4f}")
    
    # 计算总体MCRMSE
    mcrmse = np.mean(target_scores)
    LOGGER.info(f"总体 MCRMSE: {mcrmse:.4f}")
    LOGGER.info(f"各目标RMSE: {target_scores}")
    LOGGER.info(f"各折MCRMSE: {fold_scores}, 平均: {np.mean(fold_scores):.4f}")
    
    return predictions, mcrmse, target_scores, oof_predictions

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
    
    # 设置是否只使用本地文件
    if args.local_files_only:
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        CFG.local_files_only = True
        LOGGER.info("设置为离线模式：只使用本地文件")
    else:
        CFG.local_files_only = False
    
    # 设置数据目录（如果命令行提供）
    if args.data_dir:
        CFG.DATA_DIR = args.data_dir
    
    # 处理 Kaggle 环境
    if '/kaggle/' in os.path.abspath(__file__):
        # 如果在 Kaggle 环境下
        LOGGER.info("检测到Kaggle环境")
        if not args.data_dir and not os.path.exists(CFG.DATA_DIR):
            # 尝试找到竞赛数据集
            if os.path.exists('/kaggle/input/feedback-prize-english-language-learning'):
                CFG.DATA_DIR = '/kaggle/input/feedback-prize-english-language-learning'
                LOGGER.info(f"Kaggle 环境: 自动设置数据目录为 {CFG.DATA_DIR}")
    
    # 创建按模型名称组织的目录结构
    model_name_safe = CFG.model_name.replace('/', '-')
    CFG.MODEL_OUTPUT_DIR = os.path.join(CFG.OUTPUT_DIR, model_name_safe)
    
    # 设置模型目录和输出目录（分开处理）
    if args.model_dir:
        CFG.MODEL_DIR = args.model_dir
    else:
        # 如果未指定，使用模型名称子目录下的models路径
        CFG.MODEL_DIR = os.path.join(CFG.MODEL_OUTPUT_DIR, 'models')
    
    # 设置输出目录
    if args.output_dir:
        CFG.OUTPUT_DIR = args.output_dir
        
    if args.num_folds:
        CFG.num_folds = args.num_folds
    
    # 设置tokenizer目录
    if args.tokenizer_dir:
        # 使用命令行指定的tokenizer目录
        CFG.tokenizer_dir = args.tokenizer_dir
    else:
        # 首先尝试从模型名称子目录下加载tokenizer
        tokenizer_dir = os.path.join(CFG.MODEL_OUTPUT_DIR, 'tokenizer')
        if os.path.exists(tokenizer_dir):
            CFG.tokenizer_dir = tokenizer_dir
            LOGGER.info(f"找到模型子目录中的tokenizer目录: {tokenizer_dir}")
        else:
            # 尝试从模型目录旁的tokenizer目录加载（兼容旧版）
            parent_dir = os.path.dirname(CFG.MODEL_DIR)
            tokenizer_dir = os.path.join(parent_dir, 'tokenizer')
            if os.path.exists(tokenizer_dir):
                CFG.tokenizer_dir = tokenizer_dir
                LOGGER.info(f"找到模型目录旁的tokenizer目录: {tokenizer_dir}")

    # 设置配置文件路径
    if args.config_path:
        CFG.config_path = args.config_path
    else:
        # 首先尝试从模型名称子目录下加载config.pth
        config_path = os.path.join(CFG.MODEL_OUTPUT_DIR, 'config.pth')
        if os.path.exists(config_path):
            CFG.config_path = config_path
            LOGGER.info(f"找到模型子目录中的配置文件: {config_path}")
        else:
            # 尝试从模型目录的上级目录找到config.pth（兼容旧版）
            parent_dir = os.path.dirname(CFG.MODEL_DIR)
            config_path = os.path.join(parent_dir, 'config.pth')
            if os.path.exists(config_path):
                CFG.config_path = config_path
                LOGGER.info(f"找到模型目录旁的配置文件: {config_path}")
    
    # 设置池化类型
    if args.pooling_type:
        CFG.pooling_type = args.pooling_type
        LOGGER.info(f"设置池化类型: {CFG.pooling_type}")
    
    # 在离线模式下检查配置文件
    if CFG.local_files_only and not hasattr(CFG, 'config_path'):
        LOGGER.warning("离线模式下未找到配置文件，将尝试在模型目录旁查找")
    
    # 创建输出目录
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    
    # 准备输出文件路径
    output_file_path = os.path.join(CFG.OUTPUT_DIR, args.output_file)
    
    # 设置日志
    LOGGER.info(f"============ 预测开始 ============")
    LOGGER.info(f"使用配置: {args.config if args.config != 'default' else '默认配置'}")
    LOGGER.info(f"模型: {CFG.model_name}")
    LOGGER.info(f"模型目录: {CFG.MODEL_DIR}")
    LOGGER.info(f"数据目录: {CFG.DATA_DIR}")
    LOGGER.info(f"输出目录: {CFG.OUTPUT_DIR}")
    LOGGER.info(f"tokenizer目录: {getattr(CFG, 'tokenizer_dir', '未指定')}")
    LOGGER.info(f"模型配置文件: {getattr(CFG, 'config_path', '未指定')}")
    LOGGER.info(f"批次大小: {CFG.batch_size}")
    LOGGER.info(f"最大序列长度: {CFG.max_len}")
    LOGGER.info(f"交叉验证折数: {CFG.num_folds}")
    LOGGER.info(f"池化类型: {getattr(CFG, 'pooling_type', 'mean')}")
    
    # 设置种子
    seed_everything(CFG.seed)
    
    # 导入依赖模块 - 在CFG设置完成后导入
    from src.data.dataset import TestDataset, get_test_dataloader
    from src.models.model import FeedbackModel
    
    # 加载测试数据
    test_df = pd.read_csv(os.path.join(CFG.DATA_DIR, 'test.csv'))
    submission = pd.read_csv(os.path.join(CFG.DATA_DIR, 'sample_submission.csv'))
    
    # 创建数据集实例
    test_dataset = TestDataset(CFG, test_df)
    
    # 使用数据集实例的tokenizer创建数据加载器
    test_loader = get_test_dataloader(test_dataset, CFG.batch_size, CFG.num_workers, test_dataset.tokenizer)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型

    if not args.stacking:
        model_files = []
        for file in os.listdir(CFG.MODEL_DIR):
            # 检查文件是否是模型文件（通常是.pth或.bin扩展名）
            if file.endswith('.pth') and os.path.isfile(os.path.join(CFG.MODEL_DIR, file)):
                model_path = os.path.join(CFG.MODEL_DIR, file)
                model_files.append(model_path)
                LOGGER.info(f"找到模型: {model_path}")

        if not model_files:
            LOGGER.error("未找到任何模型文件！")
            sys.exit(1)

        # 对模型文件进行排序（可选）
        model_files.sort()
        model_paths = model_files
    
    # 运行推理
    final_preds = []
    
    # 处理stacking
    if args.stacking:
        LOGGER.info(f"使用stacking进行集成")
        LOGGER.info(f"Stacking数据目录: {args.stacking_dir}")
        LOGGER.info(f"Stacking方法: {args.stacking_method}")
        LOGGER.info(f"Stacking交叉验证折数: {args.stacking_cv}")
        LOGGER.info(f"Stacking交叉验证重复次数: {args.stacking_repeats}")
        
        # 加载stacking数据
        oof_dfs, submission_dfs = load_stacking_data(args.stacking_dir)
        
        if len(oof_dfs) < 2 or len(submission_dfs) < 2:
            LOGGER.error(f"Stacking需要至少2个模型的结果，但只找到{len(oof_dfs)}个oof文件和{len(submission_dfs)}个submission文件")
            sys.exit(1)
        
        # 根据选择的方法进行stacking
        if args.stacking_method == "bayesian":
            LOGGER.info("使用贝叶斯回归进行stacking")
            final_preds, mcrmse, target_scores, oof_predictions = bayesian_stacking(oof_dfs, submission_dfs, args.stacking_cv, args.stacking_repeats)
            
            # # 保存stacking的OOF预测结果
            # stacking_oof_df = oof_dfs[0].copy()
            # for i, col in enumerate(CFG.target_cols):
            #     stacking_oof_df[f"stacking_pred_{col}"] = oof_predictions[:, i]
            
            # # 保存stacking OOF结果
            # stacking_oof_path = os.path.join(CFG.OUTPUT_DIR, f"stacking_oof_{args.stacking_method}.csv")
            # stacking_oof_df.to_csv(stacking_oof_path, index=False)
            # LOGGER.info(f"Stacking OOF预测结果已保存到: {stacking_oof_path}")
            
            # 记录stacking评估结果
            stacking_eval_path = os.path.join(CFG.OUTPUT_DIR, f"stacking_eval_{args.stacking_method}.txt")
            with open(stacking_eval_path, 'w') as f:
                f.write(f"Stacking方法: {args.stacking_method}\n")
                f.write(f"总体MCRMSE: {mcrmse:.6f}\n")
                f.write("各目标RMSE:\n")
                for i, col in enumerate(CFG.target_cols):
                    f.write(f"{col}: {target_scores[i]:.6f}\n")
            LOGGER.info(f"Stacking评估结果已保存到: {stacking_eval_path}")
            
        elif args.stacking_method == "multitask":
            LOGGER.info("使用MultiTaskLasso进行stacking")
            final_preds, mcrmse, target_scores, oof_predictions = multitask_stacking(oof_dfs, submission_dfs, args.stacking_cv, args.stacking_repeats)
            
            # # 保存stacking的OOF预测结果
            # stacking_oof_df = oof_dfs[0].copy()
            # for i, col in enumerate(CFG.target_cols):
            #     stacking_oof_df[f"stacking_pred_{col}"] = oof_predictions[:, i]
            
            # # 保存stacking OOF结果
            # stacking_oof_path = os.path.join(CFG.OUTPUT_DIR, f"stacking_oof_{args.stacking_method}.csv")
            # stacking_oof_df.to_csv(stacking_oof_path, index=False)
            # LOGGER.info(f"Stacking OOF预测结果已保存到: {stacking_oof_path}")
            
            # 记录stacking评估结果
            stacking_eval_path = os.path.join(CFG.OUTPUT_DIR, f"stacking_eval_{args.stacking_method}.txt")
            with open(stacking_eval_path, 'w') as f:
                f.write(f"Stacking方法: {args.stacking_method}\n")
                f.write(f"总体MCRMSE: {mcrmse:.6f}\n")
                f.write("各目标RMSE:\n")
                for i, col in enumerate(CFG.target_cols):
                    f.write(f"{col}: {target_scores[i]:.6f}\n")
            LOGGER.info(f"Stacking评估结果已保存到: {stacking_eval_path}")
            
        else:
            LOGGER.error(f"不支持的stacking方法: {args.stacking_method}")
            sys.exit(1)
    else:
        # 原有的模型集成逻辑
        for i, model_path in enumerate(model_paths):
            LOGGER.info(f"使用模型 {i+1}/{len(model_paths)}: {model_path}")
            
            # 初始化模型 - 传递config_path和local_files_only参数
            local_files_only = getattr(CFG, 'local_files_only', False)
            config_path = getattr(CFG, 'config_path', None)
            
            try:
                model = FeedbackModel(
                    CFG.model_name, 
                    config_path=config_path,
                    local_files_only=local_files_only,
                    pooling_type=getattr(CFG, 'pooling_type', 'mean')  # 确保使用与训练时相同的池化类型
                )
                
                # 加载模型权重
                state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
                if 'model' in state:
                    model.load_state_dict(state['model'], strict=False)  # 使用strict=False
                    LOGGER.info(f"加载模型权重成功")
                else:
                    model.load_state_dict(state, strict=False)  # 使用strict=False
                    LOGGER.info(f"加载模型权重成功")
                    
                # 运行推理
                predictions = inference_fn(test_loader, model, device)
                final_preds.append(predictions)
            except Exception as e:
                LOGGER.error(f"加载或推理模型失败: {str(e)}")
                LOGGER.error(f"如果在离线环境，请确保提供了正确的config.pth文件")
                continue
            
            # 清理内存
            torch.cuda.empty_cache()
            del model
            gc.collect()
        
        if not final_preds:
            LOGGER.error("未能成功加载任何模型进行预测！")
            sys.exit(1)
        
        # 平均所有模型的预测结果
        final_preds = np.mean(final_preds, axis=0)
    
    # 保存预测结果
    submission[CFG.target_cols] = final_preds
    submission.to_csv(output_file_path, index=False)
    LOGGER.info(f"预测结果已保存到 {output_file_path}")

if __name__ == "__main__":
    main() 