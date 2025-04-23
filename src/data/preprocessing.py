import pandas as pd
import os
from tqdm.auto import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from ..config.config import CFG
from ..utils.common import LOGGER

def load_data():
    """加载并返回训练和测试数据"""
    train = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "train.csv"))
    test = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "test.csv"))
    submission = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "sample_submission.csv"))
    
    LOGGER.info(f"train.shape: {train.shape}")
    LOGGER.info(f"test.shape: {test.shape}")
    LOGGER.info(f"submission.shape: {submission.shape}")
    
    return train, test, submission

def determine_max_len(tokenizer, texts):
    """确定最大序列长度"""
    lengths = []
    tk0 = tqdm(texts, total=len(texts))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    max_len = max(lengths) + 3  # cls & sep & sep
    LOGGER.info(f"max_len: {max_len}")
    return max_len

def prepare_folds(train, n_fold=4):
    """准备交叉验证折"""
    Fold = MultilabelStratifiedKFold(n_splits=n_fold, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.target_cols])):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)
    
    return train 