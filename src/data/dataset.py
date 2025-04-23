import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer
from ..config.config import CFG

def prepare_input(cfg, text):
    """准备模型输入"""
    inputs = cfg.tokenizer.encode_plus(
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

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label

class TestDataset(Dataset):
    """测试数据集"""
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
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

class FeedbackDataset(Dataset):
    """
    英语语言学习者的反馈评分数据集
    """
    def __init__(self, df, tokenizer_path, max_len, target_cols):
        self.df = df
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.target_cols = target_cols
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.df.iloc[index]['full_text']
        
        # 文本编码
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # 获取编码后的输入
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        token_type_ids = inputs['token_type_ids'].squeeze()
        
        # 准备标签
        if self.target_cols:
            targets = torch.tensor(self.df.iloc[index][self.target_cols].values, dtype=torch.float)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'targets': targets
            }
        else:
            # 对于测试集可能没有标签
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }

def prepare_loaders(df, fold, tokenizer_path, max_len, target_cols, batch_size, num_workers):
    """
    准备数据加载器
    
    Args:
        df: 数据集DataFrame
        fold: 当前折数
        tokenizer_path: 分词器路径
        max_len: 最大序列长度
        target_cols: 目标列名列表
        batch_size: 批次大小
        num_workers: 数据加载线程数
        
    Returns:
        训练和验证数据加载器
    """
    from torch.utils.data import DataLoader
    
    # 划分训练集和验证集
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)
    
    # 创建数据集
    train_dataset = FeedbackDataset(train_df, tokenizer_path, max_len, target_cols)
    valid_dataset = FeedbackDataset(valid_df, tokenizer_path, max_len, target_cols)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader 