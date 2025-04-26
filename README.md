# Feedback-Prize-NLP-项目

本仓库包含从Kaggle笔记本 [FB3 DeBERTa v3 Base Baseline Train](https://www.kaggle.com/code/abhishek/fb3-deberta-v3-base-baseline-train) 转换而来的结构化代码，并进行了额外的修改和优化，以提高性能和可用性。

## 项目简介

本项目是基于Kaggle比赛 "Feedback Prize - English Language Learning" 的解决方案。该比赛要求参赛者构建模型，评估英语学习者写作的六个方面：连贯性、语法、词汇、表达、语法和规范。

我们通过以下方式优化了原始笔记本的代码：
- 将代码结构化为模块化组件
- 添加了命令行参数支持
- 验证了训练和预测的代码正确性
- 增加了更多配置选项
- 支持动态加载自定义配置文件
- 集成了Weights & Biases实验跟踪功能

## 项目结构

```
Feedback-Prize-NLP-Project
├── data/                  # 数据目录（需自行添加比赛数据）
├── output/                # 模型输出目录
│   ├── models/            # 保存的模型文件
│   ├── tokenizer/         # tokenizer缓存
│   └── results/           # 预测结果
├── scripts/               # 训练和预测脚本
│   ├── train.py           # 训练模型脚本
│   ├── predict.py         # 生成预测脚本
│   └── README.md          # 脚本使用说明
├── src/                   # 源代码
│   ├── config/            # 配置模块
│   ├── data/              # 数据处理模块
│   ├── models/            # 模型定义
│   ├── training/          # 训练逻辑
│   └── utils/             # 工具函数
├── experiments/           # 实验配置文件（可选）
│   └── configs/           # 自定义配置文件
├── requirements.txt       # 项目依赖
└── README.md              # 项目说明
```

## 安装与环境配置

1. 克隆仓库：
```bash
git clone https://github.com/jitehabosmys/Feedback-Prize-NLP-Project.git
cd Feedback-Prize-NLP-Project
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. （可选）安装Weights & Biases：
```bash
pip install wandb
```

5. 下载比赛数据并放置在 `data/` 目录下

## 使用方法

### 训练模型

训练脚本(`scripts/train.py`)提供了简洁的参数设计：

#### 基本用法

训练单个折：
```bash
python scripts/train.py --fold 0
```

使用调试模式（仅使用少量数据）：
```bash
python scripts/train.py --debug
```

自定义模型和输出目录：
```bash
python scripts/train.py --model "microsoft/deberta-v3-base" --output_dir "output/deberta-base"
```

训练所有折：
```bash
python scripts/train.py --train_all_data
```

#### 高级用法

使用自定义配置文件：
```bash
python scripts/train.py --config experiments/configs/large_model_config.py
```

组合使用：
```bash
python scripts/train.py --config experiments/configs/large_model_config.py --batch_size 4 --debug
```

### 使用Weights & Biases跟踪实验

启用Weights & Biases跟踪：
```bash
python scripts/train.py --use_wandb
```

指定Wandb项目和运行名称：
```bash
python scripts/train.py --use_wandb --wandb_project "my-project" --wandb_run_name "experiment-1"
```

跟踪模型结构和参数变化：
```bash
python scripts/train.py --use_wandb --wandb_watch_model
```

### 生成预测

预测脚本(`scripts/predict.py`)用于生成测试集预测：

#### 基本用法

使用默认设置进行预测：
```bash
python scripts/predict.py
```

指定模型和批次大小：
```bash
python scripts/predict.py --model "microsoft/deberta-v3-base" --batch_size 16
```

指定模型目录和输出文件：
```bash
python scripts/predict.py --model_dir "output/deberta-base" --output_file "my_submission.csv"
```

#### 高级用法

使用自定义配置：
```bash
python scripts/predict.py --config experiments/configs/large_model_config.py
```

## 参数说明

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model | 预训练模型名称 | 配置文件中的值 |
| --batch_size | 批次大小 | 配置文件中的值 |
| --fold | 要训练的折数 | 0 |
| --train_all_data | 训练所有折 | False |
| --debug | 调试模式 | False |
| --output_dir | 输出目录 | 配置文件中的值 |
| --seed | 随机种子 | 配置文件中的值 |
| --config | 配置文件路径 | default (使用内置配置) |
| --use_wandb | 启用Weights & Biases | False |
| --wandb_project | Wandb项目名称 | feedback-prize-ell |
| --wandb_entity | Wandb组织名称 | None |
| --wandb_run_name | Wandb运行名称 | 自动生成 |
| --wandb_watch_model | 是否跟踪模型参数 | False |

### 预测参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model | 预训练模型名称 | 配置文件中的值 |
| --model_dir | 模型目录 | 配置文件中的值 |
| --output_file | 输出文件名 | submission.csv |
| --num_folds | 使用多少折模型进行集成 | 配置文件中的值 |
| --batch_size | 批次大小 | 配置文件中的值 |
| --seed | 随机种子 | 配置文件中的值 |
| --config | 配置文件路径 | default (使用内置配置) |

## 配置系统

本项目引入了灵活的配置系统，允许：

1. 使用内置默认配置
2. 通过命令行参数覆盖特定配置项
3. 使用自定义配置文件

### 创建自定义配置文件

配置文件是标准Python模块，需要包含一个名为`CFG`的对象：

```python
# experiments/configs/my_config.py
import os
import torch
from src.config.config import CFG as BaseCFG

class CFG(BaseCFG.__class__):
    # 模型配置
    model_name = 'microsoft/deberta-v3-base'
    max_len = 384
    
    # 训练配置
    batch_size = 16
    epochs = 3
    encoder_lr = 1e-5
    decoder_lr = 1e-5
    
    # Wandb配置
    use_wandb = True  # 默认启用
    wandb_project = "my-project"
```

## 实验跟踪与可视化

本项目集成了Weights & Biases (wandb) 用于实验跟踪和可视化。启用wandb后，您可以获得以下好处：

1. **训练进度可视化**：实时查看损失、指标等变化
2. **超参数跟踪**：记录并比较不同实验的超参数设置
3. **模型对比**：直观比较不同模型的性能差异
4. **远程监控**：远程观察长时间训练任务的进展
5. **结果共享**：轻松与团队分享实验结果

### 使用步骤

1. 安装wandb：`pip install wandb`
2. 注册账号：[https://wandb.ai/](https://wandb.ai/)
3. 登录wandb：`wandb login`
4. 启用wandb：`--use_wandb`参数

wandb跟踪的指标包括：
- 训练/验证损失
- 各个目标的得分
- 学习率变化
- 模型梯度
- 每折和整体CV分数

## 注意事项

1. 脚本会自动使用GPU（如果可用），否则会使用CPU
2. 训练脚本会自动保存最佳模型到指定的输出目录的`models`子目录
3. 预测脚本会自动平均所有找到的模型的预测结果
4. 预测结果会保存在输出目录的`results`子目录中
5. 配置文件能够为不同的实验提供完整和可复现的配置
6. wandb功能默认禁用，需要显式启用

## 未来工作计划

我们计划在未来的开发中添加基于 `rapids-svr-cv-0-450-lb-0-44x.ipynb` 笔记本的方法，该方法使用多个预训练模型的嵌入特征结合RAPIDS SVR进行回归预测，表现出色。

## 贡献

欢迎提交Issue和Pull Request来完善项目。

## 许可

[MIT](LICENSE) 