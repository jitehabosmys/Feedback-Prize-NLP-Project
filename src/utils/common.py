import os
import random
import numpy as np
import torch
import time
import math
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

def get_logger(filename=None):
    """返回logger对象"""
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    if filename is not None:
        handler2 = FileHandler(filename=f"{filename}.log")
        handler2.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=42):
    """设置随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    """将秒转换为分钟格式"""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    """计算经过的时间和剩余时间"""
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def collate(inputs):
    """收集并裁剪数据的辅助函数"""
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs 

# 对抗训练
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                if param.grad is None:
                    continue
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

import torch.nn.functional as F

class VAT:
    """
    Virtual Adversarial Training (VAT)

    Parameters
    ----------
    model : torch.nn.Module
        下游任务模型，需包含属性 ``backbone`` 或实现 ``get_input_embeddings``。
    xi : float, default=1e-6
        Finite‐difference 孵化噪声系数。
    epsilon : float, default=1.0
        最终对抗扰动的半径。
    ip : int, default=1
        Power‐iteration 次数；≥1 时效果更稳，1 通常已够用。
    """

    def __init__(self, model, xi: float = 1e-6, epsilon: float = 1.0, ip: int = 1):
        self.model = model
        self.xi = xi
        self.epsilon = epsilon
        self.ip = ip

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def forward_kl(self, inputs: dict, logits_clean: torch.Tensor) -> torch.Tensor:
        """
        计算 VAT KL-loss。  
        ⚠️ 仅在 embedding 空间加扰动；不会破坏 ``input_ids`` 的整型索引。

        Parameters
        ----------
        inputs : dict
            传给模型的原始 batch（含 ``input_ids``、``attention_mask`` 等）。
        logits_clean : torch.Tensor
            模型在 *clean* 输入上的输出，用于作为 KL 散度的"真"分布。

        Returns
        -------
        torch.Tensor
            单标量 KL loss。
        """
        embeddings = self._get_embeddings(inputs).detach()        # [B, L, H]
        d = torch.randn_like(embeddings)                           # 随机初始化扰动
        d = self._l2_normalize(d)

        # ---------- Power-iteration 探索最激进方向 ---------- #
        for _ in range(self.ip):
            d.requires_grad_()                                     # 重新开启梯度
            self.model.zero_grad()

            perturbed = embeddings + self.xi * d
            perturbed_inputs = self._build_perturbed_batch(inputs, perturbed)

            logits_perturbed = self.model(perturbed_inputs)
            kl = F.kl_div(
                F.log_softmax(logits_perturbed, dim=-1),
                F.softmax(logits_clean, dim=-1),
                reduction="batchmean",
            )
            kl.backward()
            d = self._l2_normalize(d.grad).detach()                # 下一步迭代的方向

        # ---------- 得到最终扰动并计算 VAT-loss ---------- #
        r_vat = self.epsilon * d
        perturbed_final = embeddings + r_vat
        final_inputs = self._build_perturbed_batch(inputs, perturbed_final)

        logits_final = self.model(final_inputs)
        vat_loss = F.kl_div(
            F.log_softmax(logits_final, dim=-1),
            F.softmax(logits_clean, dim=-1),
            reduction="batchmean",
        )
    
        return vat_loss

    # --------------------------------------------------------------------- #
    #  Helpers
    # --------------------------------------------------------------------- #
    def _get_embeddings(self, inputs: dict) -> torch.Tensor:
        """
        根据输入 batch 抽取 *word* embedding。
        支持 `model.backbone.get_input_embeddings()` 或 `model.get_input_embeddings()`.
        """
        if hasattr(self.model, "backbone"):
            embed_layer = self.model.backbone.get_input_embeddings()
        else:
            embed_layer = self.model.get_input_embeddings()

        input_ids = inputs["input_ids"]
        return embed_layer(input_ids)

    @staticmethod
    def _l2_normalize(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm = torch.norm(t.view(t.size(0), -1), dim=1, keepdim=True) + eps
        return t / norm.view(-1, 1, 1)

    @staticmethod
    def _build_perturbed_batch(orig_inputs: dict, perturbed_embeds: torch.Tensor) -> dict:
        """
        构造一个新的 `inputs` dict：
        1. 使用 `inputs_embeds` 替换 `input_ids`
        2. 其余字段（attention_mask, token_type_ids 等）保持不变
        """
        new_inputs = {k: v for k, v in orig_inputs.items() if k != "input_ids"}
        new_inputs["inputs_embeds"] = perturbed_embeds
        return new_inputs

class AWP:
    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1e-4, adv_eps=1e-2):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                if param.grad is None:
                    continue
                grad = param.grad
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    norm = torch.norm(param.data)
                    self.backup_eps[name] = self.adv_eps * norm
                norm_grad = torch.norm(grad)
                if norm_grad != 0 and not torch.isnan(norm_grad):
                    r_at = self.adv_lr * grad / (norm_grad + 1e-8) * (torch.norm(param.data) + 1e-8)
                    r_at = torch.clamp(r_at, -self.backup_eps[name], self.backup_eps[name])
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name] 