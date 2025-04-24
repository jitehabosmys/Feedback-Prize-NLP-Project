import os
import torch
import warnings
warnings.filterwarnings("ignore")

class CFG:
    # 基本设置
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    debug = False
    print_freq = 100  # 每多少步打印一次训练信息
    
    # 数据设置
    num_folds = 5  # 交叉验证折数
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    max_len = 512  # 最大序列长度
    
    # 模型设置
    model_name = 'microsoft/deberta-v3-large'  # 预训练模型
    gradient_checkpointing = True  # 启用梯度检查点以减少显存使用
    
    # 训练设置
    epochs = 5
    batch_size = 8
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    weight_decay = 0.01
    max_grad_norm = 1.0
    min_lr = 1e-6
    
    # 数据加载设置
    num_workers = 4
    
    # 数据处理配置
    eps = 1e-6
    betas = (0.9, 0.999)
    scheduler = 'cosine'
    num_warmup_steps = 0
    num_cycles = 0.5
    batch_scheduler = True
    
    # 混合精度训练
    apex = True
    gradient_accumulation_steps = 1
    
    # 文件路径
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    
    # 确保输出目录存在
    os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)

# 创建必要的目录
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output"), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output", "models"), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output", "tokenizer"), exist_ok=True) 