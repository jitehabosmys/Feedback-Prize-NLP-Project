import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from src.config.config import CFG

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
        
        # 加载预训练模型配置和模型
        config = AutoConfig.from_pretrained(model_name)
        config.update({"output_hidden_states": True})
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        
        # 设置backbone的梯度检查点以节省内存
        if CFG.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            
        # 获取隐藏层大小
        self.hidden_size = config.hidden_size
        
        # 池化层
        self.pool = MeanPooling()
        
        # 回归头，对应6个回归目标
        self.fc = nn.Linear(self.hidden_size, 6)
        
        # Dropout用于防止过拟合
        self.dropout = nn.Dropout(0.2)
        
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