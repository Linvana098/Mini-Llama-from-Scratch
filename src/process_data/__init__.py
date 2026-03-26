"""
数据处理模块：语料清洗、预处理、训练数据加载
实现自回归预训练的样本构建逻辑
"""
from .data_preprocess import preprocess_corpus
from .data_loader import PretrainDataLoader