"""
训练模块：优化器、学习率调度、训练循环
实现预训练全流程，包含loss计算、日志记录、模型保存
"""
from .optim_scheduler import get_optimizer, get_lr_scheduler
from .trainer import PretrainTrainer