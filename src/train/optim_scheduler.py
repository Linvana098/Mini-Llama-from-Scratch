import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import math

def get_optimizer(model: nn.Module, config: dict) -> Optimizer:
    """
    获取预训练优化器（AdamW）
    Args:
        model: 模型实例
        config: 训练配置（来自train_config.yaml）
            - optimizer: 优化器参数
                - lr: 学习率
                - weight_decay: 权重衰减
                - eps: 防止除0的小值
    Returns:
        优化器实例
    """
    optim_config = config["optimizer"]

    # 分离权重衰减和非权重衰减参数（Llama优化）
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 偏置和RMSNorm权重不做权重衰减
        if name.endswith(".bias") or ("norm" in name and "weight" in name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # 构建参数组
    params = [
        {"params": decay_params, "weight_decay": float(optim_config["weight_decay"])},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # 初始化AdamW优化器
    optimizer = torch.optim.AdamW(
        params,
        lr=float(optim_config["lr"]),
        eps=float(optim_config["eps"]),
    )

    print(f"初始化AdamW优化器，学习率：{optim_config['lr']}，权重衰减：{optim_config['weight_decay']}")

    return optimizer

class CosineLRSchedulerWithWarmup(LRScheduler):
    """
    带warmup的余弦学习率调度器
    逻辑：前warmup_steps线性上升，之后余弦衰减
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 eta_min: float = 0.0,
                 last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: 优化器实例
            warmup_steps: 预热步数
            total_steps: 总训练步数
            eta_min: 最小学习率
            last_epoch: 上一轮epoch
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super(CosineLRSchedulerWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """计算当前学习率"""
        step = self.last_epoch

        # 1. Warmup阶段：线性上升
        if step < self.warmup_steps:
            lr = step / self.warmup_steps * self.base_lrs[0]
        # 2. 余弦衰减阶段
        else:
            # 剩余步数
            remaining_steps = self.total_steps - self.warmup_steps
            # 当前衰减步数
            decay_step = step - self.warmup_steps
            # 余弦衰减公式
            lr = self.eta_min + 0.5 * (self.base_lrs[0] - self.eta_min) * (
                1 + torch.cos(torch.tensor(decay_step / remaining_steps * math.pi))
            )

        return [lr for _ in self.base_lrs]

def get_lr_scheduler(optimizer: Optimizer, config: dict) -> LRScheduler:
    """
    获取学习率调度器
    Args:
        optimizer: 优化器实例
        config: 训练配置（来自train_config.yaml）
            - lr_scheduler: 调度器参数
                - name: 调度器名称
                - warmup_steps: 预热步数
                - total_steps: 总步数
    Returns:
        学习率调度器实例
    """
    scheduler_config = config["lr_scheduler"]

    if scheduler_config["name"] == "cosine":
        scheduler = CosineLRSchedulerWithWarmup(
            optimizer,
            warmup_steps=scheduler_config["warmup_steps"],
            total_steps=scheduler_config["total_steps"],
            eta_min=0.0,
        )
    else:
        # 默认为常量学习率
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    print(f"初始化学习率调度器：{scheduler_config['name']}，warmup步数：{scheduler_config['warmup_steps']}")

    return scheduler