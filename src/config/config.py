import os
import torch
import warnings
warnings.filterwarnings("ignore")

class CFG:
    # 基本设置
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    debug = False
    print_freq = 20  # 每多少步打印一次训练信息
    
    # 数据设置
    num_folds = 4  # 交叉验证折数
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    max_len = 512  # 最大序列长度
    
    # 模型设置
    model_name = 'microsoft/deberta-v3-base'  # 预训练模型
    gradient_checkpointing = True  # 启用梯度检查点以减少显存使用
    
    # 训练设置
    epochs = 4
    batch_size = 8
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    weight_decay = 0.01
    max_grad_norm = 1000
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
    
    # 在 Kaggle 环境中自动使用 /kaggle/working/
    if '/kaggle/' in ROOT_DIR and '/kaggle/input/' in ROOT_DIR:
        OUTPUT_DIR = '/kaggle/working/'
        # 如果在 Kaggle 环境中 DATA_DIR 也可能需要调整
        if os.path.exists('/kaggle/input/feedback-prize-english-language-learning'):
            DATA_DIR = '/kaggle/input/feedback-prize-english-language-learning'
        else:
            DATA_DIR = os.path.join(ROOT_DIR, 'data')
    else:
        OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
        DATA_DIR = os.path.join(ROOT_DIR, 'data')
    
    # 确保在非 Kaggle 环境或可写目录中创建文件夹
    if not ('/kaggle/input/' in OUTPUT_DIR):
        os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
    
    # Wandb配置
    use_wandb = False  # 默认禁用
    wandb_project = "feedback-prize-ell"  # 项目名称
    wandb_entity = None  # 组织名称，默认为个人账户
    wandb_run_name = None  # 运行名称，为None时自动生成
    wandb_log_interval = 10  # 日志记录间隔（步数）
    wandb_watch_model = False  # 是否使用wandb watch跟踪模型

# 创建必要的目录（仅在非 Kaggle 只读环境中执行）
if not ('/kaggle/input/' in os.path.dirname(os.path.dirname(os.path.dirname(__file__)))):
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output", "models"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output", "tokenizer"), exist_ok=True) 