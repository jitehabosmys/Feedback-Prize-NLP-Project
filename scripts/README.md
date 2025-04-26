# 训练和预测脚本使用说明

这个目录包含了用于模型训练和预测的脚本，从原始的Jupyter笔记本转换而来并进行了优化。

## 环境准备

在使用这些脚本前，请确保你已经安装了所有必要的依赖：

```bash
pip install -r ../requirements.txt
```

如果您想使用Weights & Biases进行实验跟踪，还需要安装：

```bash
pip install wandb
```

## 数据准备

请确保在`../data/`目录下有以下文件：
- `train.csv`：训练数据
- `test.csv`：测试数据
- `sample_submission.csv`：提交样例

## 训练脚本使用

训练脚本(`train.py`)提供了简化的参数设计：

```bash
python train.py --help
```

### 基本用法

训练单个折：
```bash
python train.py --fold 0
```

使用调试模式（仅使用少量数据）：
```bash
python train.py --debug
```

自定义模型和输出目录：
```bash
python train.py --model "microsoft/deberta-v3-base" --output_dir "../output/deberta-base"
```

训练所有折：
```bash
python train.py --train_all_data
```

### 高级用法

使用自定义配置文件：
```bash
python train.py --config ../experiments/configs/large_model_config.py
```

组合使用：
```bash
python train.py --config ../experiments/configs/large_model_config.py --batch_size 4 --debug
```

### 使用Weights & Biases跟踪实验

启用Weights & Biases跟踪：
```bash
python train.py --use_wandb
```

指定Wandb项目和运行名称：
```bash
python train.py --use_wandb --wandb_project "my-project" --wandb_run_name "experiment-1"
```

跟踪模型结构和参数变化：
```bash
python train.py --use_wandb --wandb_watch_model
```

注意：首次使用wandb需要先登录：`wandb login`

### 主要参数

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

## 预测脚本使用

预测脚本(`predict.py`)用于生成测试集预测：

```bash
python predict.py --help
```

### 基本用法

使用默认设置进行预测：
```bash
python predict.py
```

指定模型和批次大小：
```bash
python predict.py --model "microsoft/deberta-v3-base" --batch_size 16
```

指定模型目录和输出文件：
```bash
python predict.py --model_dir "../output/deberta-base" --output_file "my_submission.csv"
```

### 高级用法

使用自定义配置：
```bash
python predict.py --config ../experiments/configs/large_model_config.py
```

### 主要参数

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

1. 使用内置默认配置 (`src/config/config.py`)
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
    use_wandb = True  # 默认启用Wandb
    wandb_project = "my-feedback-project"
```

## 实验跟踪与可视化

Weights & Biases (wandb) 提供了强大的实验跟踪功能。启用后，可以跟踪以下指标：

1. **训练过程**：实时观察训练和验证损失
2. **模型性能**：每个目标的评分和整体得分
3. **超参数**：记录并比较不同配置的效果
4. **资源使用**：GPU内存和CPU使用情况
5. **可视化**：自动生成训练曲线和比较图表

### Wandb集成选项

- **训练步骤日志**：每隔`wandb_log_interval`步记录一次训练指标
- **每个epoch记录**：每个epoch结束后记录验证指标
- **最佳模型跟踪**：记录最佳模型的epoch和得分
- **模型参数跟踪**：通过`--wandb_watch_model`启用模型参数可视化

## 输出目录结构

脚本输出将保存在指定的输出目录（默认为`../output`）下，结构如下：

```
output/
├── models/             # 保存的模型文件
│   └── model-name_fold0_best.pth
├── tokenizer/          # tokenizer缓存目录
├── results/            # 预测结果
│   └── submission.csv
└── oof_df.csv          # 交叉验证结果
```

## 注意事项

1. 这些脚本会自动使用GPU（如果可用），否则会使用CPU
2. 训练脚本会自动保存最佳模型到指定的输出目录的`models`子目录
3. 预测脚本会自动平均所有找到的模型的预测结果
4. 预测结果会保存在输出目录的`results`子目录中
5. tokenizer和模型缓存都将保存在项目目录中，而不是默认的系统缓存目录
6. Wandb功能默认禁用，需要通过参数或配置文件启用 