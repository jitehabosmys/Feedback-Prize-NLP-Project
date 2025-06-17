# Feedback-Prize-ELL 项目

本仓库包含从Kaggle笔记本 [FB3 DeBERTa v3 Base Baseline Train](https://www.kaggle.com/code/abhishek/fb3-deberta-v3-base-baseline-train) 转换而来的结构化代码，并进行了一系列优化与扩展，以提高性能和可用性。

## 项目简介

本项目基于Kaggle比赛 "Feedback Prize - English Language Learning"，目标是构建模型评估英语学习者写作的六个方面：连贯性、语法、词汇、表达、语法和规范。

我们的主要改进包括：
- 结构化、模块化的代码架构
- 多种优化策略的系统性实现
- 对抗训练与GNN标签图的集成
- 可配置的训练与推理流程

## 项目结构

```
Feedback-Prize-ELL
├── data/                  # 数据目录
├── output/                # 模型输出目录
│   ├── models/            # 保存的模型文件
│   └── tokenizer/         # tokenizer缓存
├── scripts/               # 训练和预测脚本
│   ├── train.py           # 训练模型脚本
│   └── predict.py         # 生成预测脚本
├── src/                   # 源代码
│   ├── config/            # 配置模块
│   ├── data/              # 数据处理模块
│   ├── models/            # 模型定义
│   ├── training/          # 训练逻辑
│   └── utils/             # 工具函数
└── requirements.txt       # 项目依赖
```

## 安装与环境配置

1. 克隆仓库并进入项目目录
2. 创建并激活虚拟环境（推荐）
3. 安装依赖：`pip install -r requirements.txt`

## 优化策略

我们实现了多种优化策略来提升基线模型性能：

### 1. 模型初始化
支持多种初始化策略，包括随机正态分布（默认）、Kaiming初始化、正交初始化等。

### 2. 多种池化方法
实现了不同的池化策略以聚合token表示：
- 平均池化（默认）
- CLS池化
- 注意力池化
- 加权层池化

### 3. 增强型损失函数
基于Smooth L1 Loss，并支持辅助损失组件：
- Pearson相关系数损失
- 排序损失
- 组合损失

### 4. 层级学习率衰减
实现了分层学习率衰减（LLRD），允许对预训练模型的不同层使用不同的学习率，以更好地保留预训练知识。

### 5. 对抗训练
支持多种对抗训练策略来提高模型鲁棒性：
- Fast Gradient Method (FGM)
- Virtual Adversarial Training (VAT)
- Adversarial Weight Perturbation (AWP)

### 6. 标签图建模
使用图卷积网络（GCN）来建模六个目标维度间的关系，利用标签相关性改善预测性能。

## 使用方法

### 训练模型

```bash
# 基本训练（单折）
python scripts/train.py --fold 0

# 使用GNN标签图
python scripts/train.py --fold 0 --use_gnn

# 使用FGM对抗训练
python scripts/train.py --fold 0 --use_fgm --fgm_epsilon 0.5

# 使用多种优化策略
python scripts/train.py --fold 0 --pooling_type cls --init_type orthogonal \
                       --layerwise_lr_decay 0.9 --loss_type pearson \
                       --pearson_loss_weight 0.1
```

### 生成预测

```bash
# 使用默认设置进行预测
python scripts/predict.py

# 指定模型目录
python scripts/predict.py --model_dir "output/deberta-v3-base/models"
```

## 主要参数

| 参数 | 说明 |
|------|------|
| --model | 预训练模型名称 |
| --pooling_type | 池化策略：mean, cls, attention, weighted_layer |
| --init_type | 初始化策略：normal, xavier_uniform, kaiming_normal, orthogonal等 |
| --loss_type | 损失函数类型：l1, mse, log_cosh, pearson, rank |
| --use_gnn | 启用GNN标签图 |
| --use_fgm | 使用FGM对抗训练 |
| --use_vat | 使用VAT对抗训练 |
| --use_awp | 使用AWP对抗训练 |
| --layerwise_lr_decay | 分层学习率衰减率 |

## 实验结果

我们的优化提升了基线模型性能：

| 策略 | CV得分 | LB得分 |
|------|--------|--------|
| 原始基线 | 0.4540 | 0.4419 |
| 复现结果 | 0.4547 | 0.4430 |
| 正交初始化 | 0.4525 | 0.4408 |
| CLS池化 | 0.4555 | 0.4415 |
| L1+Pearson损失 | 0.4530 | 0.4421 |
| LLRD (0.9) | 0.4540 | 0.4425 |
| 最佳单模型 | 0.4514 | 0.4393 | 