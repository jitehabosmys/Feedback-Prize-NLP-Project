import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer
from ..config.config import CFG
import os

# 全局tokenizer缓存
_TOKENIZER_CACHE = {}

def get_tokenizer(model_name):
    """获取tokenizer，如果已加载则从缓存返回"""
    if model_name not in _TOKENIZER_CACHE:
        print(f"加载tokenizer: {model_name}")
        # 获取项目根目录下的output/tokenizer路径作为缓存目录
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output", "tokenizer")
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return _TOKENIZER_CACHE[model_name]

def prepare_input(cfg, text, tokenizer):
    """准备模型输入"""
    inputs = tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=cfg.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs

class TrainDataset(Dataset):
    """训练数据集"""
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.labels = df[cfg.target_cols].values
        self.tokenizer = get_tokenizer(cfg.model_name)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item], self.tokenizer)
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label

class TestDataset(Dataset):
    """测试数据集"""
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.tokenizer = get_tokenizer(cfg.model_name)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item], self.tokenizer)
        return inputs

def get_train_dataloader(train_dataset, batch_size, num_workers):
    """获取训练数据加载器"""
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

def get_valid_dataloader(valid_dataset, batch_size, num_workers):
    """获取验证数据加载器"""
    return DataLoader(
        valid_dataset,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

def get_test_dataloader(test_dataset, batch_size, num_workers, tokenizer):
    """获取测试数据加载器"""
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding='longest'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    ) 