"""
Mini-Llama项目核心代码包
包含模型架构、数据处理、训练、生成全流程
"""
__version__ = "1.0.3"
__author__ = "LIN, Zhihao"

# 导出核心模块，方便外部调用
from .model import decoder_model
from .train import trainer
from .generate import text_generator