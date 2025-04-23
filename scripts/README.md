# 训练和预测脚本使用说明

这个目录包含了用于模型训练和预测的脚本，从原始的Jupyter笔记本转换而来。

## 环境准备

在使用这些脚本前，请确保你已经安装了所有必要的依赖：

```bash
pip install -r ../requirements.txt
```

## 数据准备

请确保在`../data/`目录下有以下文件：
- `train.csv`：训练数据
- `test.csv`：测试数据
- `sample_submission.csv`：提交样例

## 训练脚本使用

训练脚本(`train.py`)提供了多种参数来自定义训练过程：

```bash
python train.py --help
```

### 基本用法

训练单个折：
```bash
python train.py --fold 0 --batch_size 8 --epochs 5
```

使用调试模式（仅使用少量数据）：
```bash
python train.py --debug
```

自定义模型和输出目录：
```bash
python train.py --model "microsoft/deberta-v3-large" --output_dir "../output/deberta-large"
```

训练所有折：
```bash
python train.py --train_all_data
```

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

指定模型目录和批次大小：
```bash
python predict.py --model_dir "../output" --batch_size 16
```

调试模式：
```bash
python predict.py --debug
```

## 注意事项

1. 这些脚本会自动使用GPU（如果可用），否则会使用CPU
2. 训练脚本会自动保存最佳模型到指定的输出目录
3. 预测脚本会自动平均所有找到的模型的预测结果
4. 最终结果会保存为`submission.csv` 