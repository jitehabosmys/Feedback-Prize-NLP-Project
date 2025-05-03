import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer
from ..config.config import CFG
import os

def get_tokenizer(model_name, tokenizer_dir=None, local_files_only=False):
    """获取tokenizer，每次返回新实例，避免潜在状态泄露"""
    print(f"加载tokenizer: {model_name}")
    
    # 优先检查指定的tokenizer目录
    if tokenizer_dir and os.path.exists(tokenizer_dir):
        print(f"使用指定目录的tokenizer: {tokenizer_dir}")
        return AutoTokenizer.from_pretrained(tokenizer_dir)
    
    # 检查模型目录旁的tokenizer目录（预测时常用）
    if '/kaggle/' in os.path.abspath(__file__) or local_files_only:
        # Kaggle环境或离线模式，尝试从本地加载
        print(f"在离线模式下加载tokenizer，使用local_files_only=True")
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    else:
        # 在线模式，从网络加载
        print(f"从网络加载tokenizer: {model_name}")
        return AutoTokenizer.from_pretrained(model_name)

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
        self.tokenizer = get_tokenizer(cfg.model_name, 
                                      tokenizer_dir=getattr(cfg, 'tokenizer_dir', None),
                                      local_files_only=getattr(cfg, 'local_files_only', False))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = self.tokenizer.encode_plus(
            self.texts[item],
            truncation=True,
            max_length=self.cfg.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
        }, torch.tensor(self.labels[item], dtype=torch.float)

class TestDataset(Dataset):
    """测试数据集"""
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.tokenizer = get_tokenizer(cfg.model_name, 
                                      tokenizer_dir=getattr(cfg, 'tokenizer_dir', None),
                                      local_files_only=getattr(cfg, 'local_files_only', False))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item], self.tokenizer)
        return inputs

def get_train_dataloader(dataset, batch_size, num_workers=4):
    """获取训练数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

def get_valid_dataloader(dataset, batch_size, num_workers=4):
    """获取验证数据加载器"""
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
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