import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from src.config.config import CFG
import os

# 全局模型缓存
_MODEL_CACHE = {}
_CONFIG_CACHE = {}

def get_pretrained_model(model_name):
    """获取预训练模型，如果已加载则从缓存返回"""
    global _MODEL_CACHE, _CONFIG_CACHE
    
    # 获取项目根目录下的output/models路径作为缓存目录
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output", "models")
    
    if model_name not in _CONFIG_CACHE:
        print(f"加载模型配置: {model_name}")
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        config.update({"output_hidden_states": True})
        _CONFIG_CACHE[model_name] = config
    else:
        config = _CONFIG_CACHE[model_name]
    
    if model_name not in _MODEL_CACHE:
        print(f"加载预训练模型: {model_name}")
        model = AutoModel.from_pretrained(model_name, config=config, cache_dir=cache_dir)
        if CFG.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        _MODEL_CACHE[model_name] = model
    
    # 直接返回缓存的引用，避免复制开销
    return _MODEL_CACHE[model_name], config

class MeanPooling(nn.Module):
    """平均池化层"""
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class FeedbackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedbackModel, self).__init__()
        
        # 直接使用缓存模型，避免重复加载
        self.backbone, config = get_pretrained_model(model_name)
            
        # 获取隐藏层大小
        self.hidden_size = config.hidden_size
        
        # 池化层
        self.pool = MeanPooling()
        
        # 回归头，对应6个回归目标
        self.fc = nn.Linear(self.hidden_size, 6)
        
        # 应用权重初始化
        self._init_weights(self.fc)
        
        # Dropout用于防止过拟合 - 与原始笔记本保持一致，不使用dropout
        self.dropout = nn.Dropout(0.0)
        
    def _init_weights(self, module):
        """
        与原始Kaggle笔记本保持一致的权重初始化方法
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.backbone.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.backbone.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.backbone(**inputs)
        last_hidden_states = outputs.last_hidden_state
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature
    
    def forward(self, inputs):
        feature = self.feature(inputs)
        feature = self.dropout(feature)
        output = self.fc(feature)
        return output 