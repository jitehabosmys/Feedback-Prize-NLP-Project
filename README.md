# Feedback-Prize-NLP-项目

本仓库包含从Kaggle笔记本 [FB3 DeBERTa v3 Base Baseline Train](https://www.kaggle.com/code/abhishek/fb3-deberta-v3-base-baseline-train) 转换而来的结构化代码，并进行了额外的修改和优化，以提高性能和可用性。

## 项目简介

本项目是基于Kaggle比赛 "Feedback Prize - English Language Learning" 的解决方案。该比赛要求参赛者构建模型，评估英语学习者写作的六个方面：连贯性、语法、词汇、表达、语法和规范。

我们通过以下方式优化了原始笔记本的代码：
- 将代码结构化为模块化组件
- 添加了命令行参数支持
- 验证了训练和预测的代码正确性。
- 增加了更多配置选项

## 项目结构

```
Feedback-Prize-NLP-Project
├── data/                  # 数据目录（需自行添加比赛数据）
├── output/                # 模型输出目录
├── scripts/               # 训练和预测脚本
│   ├── train.py           # 训练模型脚本
│   ├── predict.py         # 生成预测脚本
│   ├── README.md          # 脚本使用说明
│   └── submission.csv     # 生成的提交文件
├── src/                   # 源代码
│   ├── config/            # 配置模块
│   ├── data/              # 数据处理模块
│   ├── models/            # 模型定义
│   ├── training/          # 训练逻辑
│   └── utils/             # 工具函数
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

4. 下载比赛数据并放置在 `data/` 目录下

## 使用方法

### 训练模型

训练脚本(`scripts/train.py`)提供了多种参数来自定义训练过程：

#### 基本用法

训练单个折：
```bash
python scripts/train.py --fold 0 --batch_size 8 --epochs 5
```

使用调试模式（仅使用少量数据）：
```bash
python scripts/train.py --debug
```

自定义模型和输出目录：
```bash
python scripts/train.py --model "microsoft/deberta-v3-large" --output_dir "output/deberta-large"
```

训练所有折：
```bash
python scripts/train.py --train_all_data
```

#### 高级用法

指定交叉验证折数和学习率：
```bash
python scripts/train.py --num_folds 10 --encoder_lr 1e-5 --decoder_lr 1e-4
```

调整序列长度和批次大小：
```bash
python scripts/train.py --max_len 256 --batch_size 16
```

自定义学习率调度器：
```bash
python scripts/train.py --scheduler linear --warmup_steps 100
```

### 生成预测

预测脚本(`scripts/predict.py`)用于生成测试集预测：

#### 基本用法

使用默认设置进行预测：
```bash
python scripts/predict.py
```

指定模型目录和批次大小：
```bash
python scripts/predict.py --model_dir "output" --batch_size 16
```

调试模式：
```bash
python scripts/predict.py --debug
```

#### 高级用法

指定输出文件名和模型：
```bash
python scripts/predict.py --output_file "my_submission.csv" --model "microsoft/deberta-v3-large"
```

使用测试时增强：
```bash
python scripts/predict.py --use_tta
```

## 参数说明

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model | 预训练模型名称 | microsoft/deberta-v3-large |
| --batch_size | 批次大小 | 8 |
| --epochs | 训练轮数 | 5 |
| --encoder_lr | 编码器学习率 | 2e-5 |
| --decoder_lr | 解码器学习率 | 2e-5 |
| --max_len | 最大序列长度 | 512 |
| --num_folds | 交叉验证折数 | 5 |
| --fold | 训练单折时指定的折 | 0 |
| --scheduler | 学习率调度器类型 | cosine |
| --output_dir | 输出目录 | output |

更多参数请使用 `--help` 选项查看。

## 注意事项

1. 这些脚本会自动使用GPU（如果可用），否则会使用CPU
2. 训练脚本会自动保存最佳模型到指定的输出目录
3. 预测脚本会自动平均所有找到的模型的预测结果
4. 最终结果会保存为`submission.csv`

## 未来工作计划

我们计划在未来的开发中添加基于 `rapids-svr-cv-0-450-lb-0-44x.ipynb` 笔记本的方法，该方法：

1. 使用多个预训练模型（包括不同规模的DeBERTa模型）来提取文本嵌入
2. 将多个模型的嵌入特征连接起来
3. 使用RAPIDS SVR进行回归预测
4. 无需微调预训练模型，直接使用嵌入特征训练SVR
5. 这种方法在Kaggle竞赛中取得了很好的表现（CV: 0.450，LB: 0.44x）

未来的开发将重点关注：
- 实现多模型嵌入特征提取
- 添加RAPIDS支持（GPU加速的SVR训练）
- 优化多模型集成方法
- 提高预测性能

## 贡献

欢迎提交Issue和Pull Request来完善项目。

## 许可

[MIT](LICENSE) 