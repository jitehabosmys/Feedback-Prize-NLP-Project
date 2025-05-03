import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from ..config.config import CFG
import os
import json

def get_pretrained_model(model_name, config_path=None, local_files_only=False):
    """获取预训练模型"""
    # 优先检查是否提供了配置文件路径
    if config_path and os.path.exists(config_path):
        print(f"使用本地配置文件: {config_path}")
        try:
            # 尝试加载配置
            config = torch.load(config_path)
            
            # 添加必要的配置，确保与训练时一致
            config.update({"output_hidden_states": True})
            config.hidden_dropout = 0.
            config.hidden_dropout_prob = 0.
            config.attention_dropout = 0.
            config.attention_probs_dropout_prob = 0.
            
            print(f"使用配置创建模型（无需预训练权重）")
            # 关键修改：使用from_config而不是from_pretrained
            model = AutoModel.from_config(config)
            return model, config
            
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            print("尝试其他加载方式...")
    
    # 如果未提供配置文件，但要求使用本地文件，则查找可能的路径
    if local_files_only:
        # 检查模型目录旁是否有config.pth
        if model_name.startswith('/'):  # 绝对路径
            model_dir = os.path.dirname(model_name)
            parent_dir = os.path.dirname(model_dir)
            config_path = os.path.join(parent_dir, 'config.pth')
            if os.path.exists(config_path):
                print(f"找到本地配置文件: {config_path}")
                try:
                    config = torch.load(config_path)
                    
                    # 添加必要的配置
                    config.update({"output_hidden_states": True})
                    config.hidden_dropout = 0.
                    config.hidden_dropout_prob = 0.
                    config.attention_dropout = 0.
                    config.attention_probs_dropout_prob = 0.
                    
                    print(f"使用配置创建模型（无需预训练权重）")
                    # 使用from_config代替from_pretrained
                    model = AutoModel.from_config(config)
                    return model, config
                except Exception as e:
                    print(f"加载配置文件失败: {str(e)}")
        
        # 如果仍未找到配置文件，则报错
        raise ValueError("离线模式下必须提供配置文件(config.pth)或在模型目录旁能找到config.pth")
    
    # 在线模式，从网络下载配置和模型
    print(f"加载模型配置: {model_name}")
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output", "models")
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    config.update({"output_hidden_states": True})
    # 明确禁用所有dropout层 - 与原始笔记本保持一致
    config.hidden_dropout = 0.
    config.hidden_dropout_prob = 0.
    config.attention_dropout = 0.
    config.attention_probs_dropout_prob = 0.
    
    print(f"加载预训练模型: {model_name}")
    model = AutoModel.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    if CFG.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model, config

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
    def __init__(self, model_name, config_path=None, local_files_only=False):
        super(FeedbackModel, self).__init__()
        
        # 加载预训练模型，优先使用指定的配置文件
        self.backbone, config = get_pretrained_model(model_name, config_path=config_path, local_files_only=local_files_only)
            
        # 获取隐藏层大小
        self.hidden_size = config.hidden_size
        
        # 池化层
        self.pool = MeanPooling()
        
        # 回归头，对应6个回归目标
        self.fc = nn.Linear(self.hidden_size, 6)
        
        # 应用权重初始化
        self._init_weights(self.fc)
        
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
        # 移除dropout使用，直接将特征传递给全连接层
        output = self.fc(feature)
        return output 