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
│   ├── config.pth         # 模型配置文件（离线环境必需）
│   └── results/           # 预测结果
├── scripts/               # 训练和预测脚本
│   ├── train.py           # 训练模型脚本
│   └── predict.py         # 生成预测脚本
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

## 脚本使用说明

`scripts`目录包含了从原始Jupyter笔记本转换而来并进行优化的训练和预测脚本。在使用这些脚本前，请确保数据目录中包含必要的比赛数据文件：
- `data/train.csv`：训练数据
- `data/test.csv`：测试数据
- `data/sample_submission.csv`：提交样例

如果之前没有登录过Weights & Biases，且打算使用该功能，需要先执行：
```bash
wandb login
```

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
python scripts/predict.py --model_dir "output/deberta-base/models" --output_file "my_submission.csv"
```

#### 高级用法

使用自定义配置：
```bash
python scripts/predict.py --config experiments/configs/large_model_config.py
```

## 复现与Kaggle离线环境运行

本项目设计为能够在Kaggle离线环境中正常运行。在训练时，会自动保存两个关键文件：

1. **tokenizer** - 保存在 `output/tokenizer` 目录
2. **config.pth** - 保存在 `output` 目录

这两个文件对于离线环境至关重要，因为它们允许模型在没有网络连接的情况下加载和运行。

### 运行步骤

1. **训练模型**（在本地或Kaggle笔记本中）:
   ```bash
   python scripts/train.py --model "microsoft/deberta-v3-base" --batch_size 8 --epochs 4 --train_all_data --output_dir "output/deberta-v3-base"
   ```

2. **将训练结果保存为Kaggle数据集**:
   - 将整个 `output/deberta-v3-base` 目录（包含models、tokenizer和config.pth）保存为Kaggle数据集
   - 确保保持原始目录结构不变

3. **在Kaggle推理笔记本中运行**:
   ```bash
   !python /kaggle/input/feedback-prize-nlp-project/scripts/predict.py \
   --model "microsoft/deberta-v3-base" \
   --model_dir "/kaggle/input/your-trained-models/output/deberta-v3-base/models/" \
   --config_path "/kaggle/input/your-trained-models/output/deberta-v3-base/config.pth" \
   --output_dir "/kaggle/working/" \
   --local_files_only
   ```

### 关键参数

| 参数 | 说明 | 示例 |
|------|------|------|
| --model_dir | 指向训练好的模型文件夹 | "/kaggle/input/your-trained-models/output/deberta-v3-base/models/" |
| --config_path | 指向保存的config.pth文件 | "/kaggle/input/your-trained-models/output/deberta-v3-base/config.pth" |
| --tokenizer_dir | 指向tokenizer目录（通常自动检测） | "/kaggle/input/your-trained-models/output/deberta-v3-base/tokenizer" |
| --local_files_only | 仅使用本地文件，不尝试下载 | --local_files_only |

### 自动检测

系统会自动尝试检测以下文件：
1. 如果未指定`config_path`，会在模型目录的上一级查找`config.pth`
2. 如果未指定`tokenizer_dir`，会在模型目录的同级目录中查找`tokenizer`文件夹

为确保在Kaggle环境中正常运行，建议上传完整的输出目录结构，保持原有层次关系。

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
| --print_freq | 训练过程中打印频率 | 配置文件中的值 |
| --epochs | 训练轮次 | 配置文件中的值 |
| --max_grad_norm | 梯度裁剪阈值 | 配置文件中的值 |
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
| --tokenizer_dir | tokenizer目录 | None |
| --config_path | 模型配置文件路径 | None |
| --local_files_only | 仅使用本地文件 | False |
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

## 输出目录结构

训练和预测脚本的输出将保存在指定的输出目录（默认为`output`）下，结构如下：

```
output/
├── models/             # 保存的模型文件
│   └── model-name_fold0_best.pth
├── tokenizer/          # tokenizer文件
├── config.pth          # 模型配置文件（离线必需）
├── results/            # 预测结果
│   └── submission.csv
└── oof_df.csv          # 交叉验证结果
```

## 注意事项

1. 脚本会自动使用GPU（如果可用），否则会使用CPU
2. 训练脚本会自动保存最佳模型到指定的输出目录的`models`子目录
3. 训练脚本会自动保存tokenizer到输出目录的`tokenizer`子目录
4. 训练脚本会自动保存模型配置到输出目录的`config.pth`文件
5. 预测脚本会自动从模型目录旁的`tokenizer`目录和`config.pth`文件加载必要组件
6. 预测脚本会自动平均所有找到的模型的预测结果
7. 在离线环境（如Kaggle推理）中运行时，必须设置`--local_files_only`并确保能找到config.pth
8. wandb功能默认禁用，需要显式启用
9. 梯度裁剪实现采用了原始笔记本的方式（先裁剪再缩放），推荐的裁剪阈值：
    - 1000
    - 5000

### Wandb集成选项

使用Weights & Biases时，可以通过以下方式自定义日志记录行为：

- **训练步骤日志**：每隔`wandb_log_interval`步记录一次训练指标
- **每个epoch记录**：每个epoch结束后记录验证指标
- **最佳模型跟踪**：记录最佳模型的epoch和得分
- **模型参数跟踪**：通过`--wandb_watch_model`参数启用模型参数可视化

## 未来工作计划

我们计划在未来的开发中添加基于 `rapids-svr-cv-0-450-lb-0-44x.ipynb` 笔记本的方法，该方法使用多个预训练模型的嵌入特征结合RAPIDS SVR进行回归预测，表现出色。

## 贡献

欢迎提交Issue和Pull Request来完善项目。

## 许可

[MIT](LICENSE) 