import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from ..config.config import CFG
import os
import json
import torch.nn.functional as F

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
        # 尝试在多个可能的位置查找config.pth
        model_name_safe = model_name.replace('/', '-')
        
        # 可能的config.pth路径列表
        possible_paths = []
        
        # 如果model_name是路径（以/开头），检查其目录结构
        if model_name.startswith('/'):
            model_dir = os.path.dirname(model_name)
            parent_dir = os.path.dirname(model_dir)
            possible_paths.append(os.path.join(parent_dir, 'config.pth'))
        
        # 尝试在不同位置查找config.pth
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(root_dir, 'output')
        
        possible_paths.extend([
            os.path.join(output_dir, model_name_safe, 'config.pth'),  # 新目录结构
            os.path.join(output_dir, 'config.pth'),                   # 旧目录结构
            os.path.join(os.path.dirname(output_dir), model_name_safe, 'config.pth')  # 上级目录
        ])
        
        # 尝试加载找到的config.pth
        for path in possible_paths:
            if os.path.exists(path):
                print(f"找到本地配置文件: {path}")
                try:
                    config = torch.load(path)
                    
                    # 添加必要的配置
                    config.update({"output_hidden_states": True})
                    config.hidden_dropout = 0.
                    config.hidden_dropout_prob = 0.
                    config.attention_dropout = 0.
                    config.attention_probs_dropout_prob = 0.
                    
                    # 尝试从预训练模型目录加载
                    model_dir = os.path.dirname(path)
                    pretrained_dir = os.path.join(model_dir, "pretrained_model")
                    
                    if os.path.exists(pretrained_dir) and os.path.isdir(pretrained_dir):
                        try:
                            print(f"尝试从预训练模型目录加载: {pretrained_dir}")
                            model = AutoModel.from_pretrained(pretrained_dir, config=config)
                            print(f"成功从预训练模型目录加载模型")
                            return model, config
                        except Exception as e:
                            print(f"从预训练模型目录加载失败: {str(e)}")
                            print("将尝试其他方式...")
                    
                    # 如果没有预训练模型目录或加载失败，使用配置创建模型
                    print(f"使用配置创建模型（随机初始化）")
                    model = AutoModel.from_config(config)
                    
                    # 不再尝试加载pytorch_model.bin文件，因为它可能不兼容
                    return model, config
                    
                except Exception as e:
                    print(f"加载配置文件 {path} 失败: {str(e)}")
                    print("尝试下一个路径...")
        
        # 如果仍未找到配置文件，则报错
        raise ValueError("离线模式下必须提供配置文件(config.pth)或在模型目录旁能找到config.pth。\n"
                         "请确保已经训练过该模型，或者指定正确的配置文件路径。")
    
    # 在线模式，从网络下载配置和模型
    print(f"加载模型配置: {model_name}")
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output", "models")
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    config.update({"output_hidden_states": True})
    # 明确禁用所有dropout层 
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

class ClsPooling(nn.Module):
    """CLS池化层 - 使用[CLS]标记的表示"""
    def __init__(self):
        super(ClsPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask=None):
        # 直接取第一个token的表示（CLS token）
        return last_hidden_state[:, 0, :]

class AttentionPooling(nn.Module):
    """注意力池化层"""
    def __init__(self, in_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask == 0] = float('-inf')
        w = torch.softmax(w, 1)
        context = torch.sum(w * last_hidden_state, dim=1)
        return context

class WeightedLayerPooling(nn.Module):
    """加权层池化 - 结合多层的表示"""
    def __init__(self, num_hidden_layers, layer_start=4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )
        
    def forward(self, all_hidden_states, attention_mask):
        all_layer_embedding = torch.stack(all_hidden_states)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        
        # 应用注意力掩码并计算平均池化
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(weighted_average.size()).float()
        sum_embeddings = torch.sum(weighted_average * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings

# GCN层定义
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.linear(x)
        return F.relu(x)

# 标签图上的GCN网络
try:
    from torch_geometric.nn import GCNConv

    class LabelGCN(nn.Module):
        def __init__(self, label_dim, hidden_dim=64):
            super().__init__()
            self.gcn1 = GCNConv(label_dim, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, label_dim)

        def forward(self, label_input, edge_index, edge_weight=None):
            x = self.gcn1(label_input, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.gcn2(x, edge_index, edge_weight=edge_weight)
            return x
except ImportError:
    # 如果没有安装torch_geometric，使用简单的GCN实现
    class LabelGCN(nn.Module):
        def __init__(self, label_dim, hidden_dim=64):
            super().__init__()
            self.gcn1 = GCNLayer(label_dim, hidden_dim)
            self.gcn2 = GCNLayer(hidden_dim, label_dim)
            
        def forward(self, label_input, edge_index, edge_weight=None):
            # 创建稀疏邻接矩阵
            adj = torch.zeros((label_input.size(0), label_input.size(0)), device=label_input.device)
            adj[edge_index[0], edge_index[1]] = 1 if edge_weight is None else edge_weight
            
            x = self.gcn1(label_input, adj)
            x = self.gcn2(x, adj)
            return x

class FeedbackModel(nn.Module):
    def __init__(self, model_name, config_path=None, local_files_only=False, pooling_type=None, init_type='normal', reinit_layers=None):
        super(FeedbackModel, self).__init__()
        
        # 加载预训练模型，优先使用指定的配置文件
        self.backbone, config = get_pretrained_model(model_name, config_path=config_path, local_files_only=local_files_only)
            
        # 获取隐藏层大小
        self.hidden_size = config.hidden_size
        
        # 使用配置文件中的池化类型，如果未指定则使用参数中的值
        self.pooling_type = pooling_type or CFG.pooling_type
        
        # 保存初始化类型
        self.init_type = init_type
        
        # 重新初始化顶层（如果指定）
        if reinit_layers is not None and reinit_layers > 0:
            self._reinit_top_layers(reinit_layers, init_type)
        
        # 根据池化类型选择池化层
        if self.pooling_type == 'cls':
            self.pool = ClsPooling()
        elif self.pooling_type == 'attention':
            self.pool = AttentionPooling(self.hidden_size)
        elif self.pooling_type == 'weighted_layer':
            # 获取模型层数
            num_hidden_layers = config.num_hidden_layers
            self.pool = WeightedLayerPooling(num_hidden_layers, layer_start=CFG.layer_start)
        else:  # 默认使用平均池化
            self.pool = MeanPooling()
        
        # 回归头，对应6个回归目标
        self.label_dim = 6
        self.fc = nn.Linear(self.hidden_size, self.label_dim)
        self.label_gnn = LabelGCN(label_dim=self.label_dim)  # 添加标签图GCN
        
        # 应用权重初始化
        self._init_weights(self.fc, self.init_type)
        
    def _init_weights(self, module, init_type='normal'):
        """
        权重初始化方法，支持多种初始化策略
        
        Args:
            module: 需要初始化的模块
            init_type: 初始化类型，可选 'normal', 'xavier_uniform', 'xavier_normal', 
                      'kaiming_uniform', 'kaiming_normal', 'orthogonal'
        """
        if isinstance(module, nn.Linear):
            if init_type == 'normal':
                module.weight.data.normal_(mean=0.0, std=self.backbone.config.initializer_range)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight.data)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(module.weight.data)
            elif init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight.data)
            elif init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight.data)
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(module.weight.data)
            else:
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
            
    def _reinit_top_layers(self, num_layers, init_type='normal'):
        """
        重新初始化模型顶部的n层
        
        Args:
            num_layers: 要重新初始化的层数
            init_type: 初始化类型
        """
        # 获取所有transformer层
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            layers = self.backbone.encoder.layer
            total_layers = len(layers)
            
            # 确保不超出范围
            num_layers = min(num_layers, total_layers)
            
            print(f"重新初始化最后 {num_layers}/{total_layers} 层，使用 {init_type} 初始化")
            
            # 从顶层开始重新初始化
            for i in range(total_layers - num_layers, total_layers):
                layer = layers[i]
                print(f"重新初始化第 {i+1}/{total_layers} 层")
                
                # 重新初始化每一层中的所有权重
                for module in layer.modules():
                    if isinstance(module, nn.Linear):
                        self._init_weights(module, init_type)
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.backbone(**inputs)
        
        # 对于weighted_layer_pooling，需要所有层的隐藏状态
        if self.pooling_type == 'weighted_layer':
            all_hidden_states = outputs.hidden_states
            feature = self.pool(all_hidden_states, inputs['attention_mask'])
        else:
            # 对于其他池化方法，只需要最后一层的隐藏状态
            last_hidden_states = outputs.last_hidden_state
            feature = self.pool(last_hidden_states, inputs['attention_mask'])
            
        return feature
    
    def forward(self, inputs, label_adj=None, drop_edge_prob=0.2, training=True):
        feature = self.feature(inputs)
        raw_output = self.fc(feature)
        
        if label_adj is not None:
            # 创建单位矩阵作为标签输入
            label_eye = torch.eye(self.label_dim).to(raw_output.device)
            
            # 训练时随机丢弃边
            if training and drop_edge_prob > 0:
                mask = torch.bernoulli((1 - drop_edge_prob) * torch.ones_like(label_adj)).to(label_adj.device)
                label_adj = label_adj * mask
                label_adj.fill_diagonal_(1.0)  # 确保自环总是存在
            
            # 将邻接矩阵转换为edge_index格式
            edge_index = (label_adj > 0).nonzero(as_tuple=False).T.contiguous()
            edge_weight = label_adj[edge_index[0], edge_index[1]]
            
            # 使用标签图GCN获取权重
            gat_weights = self.label_gnn(label_eye, edge_index, edge_weight=edge_weight)
            # 应用权重到输出
            output = torch.matmul(raw_output, gat_weights.T)
        else:
            output = raw_output
            
        return output 