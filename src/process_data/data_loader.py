import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class PretrainDataset(Dataset):
    """
    预训练数据集类
    实现自回归预训练的样本构建：输入[token1, token2,...n-1]，标签[token2,...n]
    """
    def __init__(self, data_path: str, pad_token_id: int):
        """
        Args:
            data_path: 处理后的token id数据路径（npy文件）
            pad_token_id: PAD token id
        """
        self.data = np.load(data_path)
        self.pad_token_id = pad_token_id

        valid_samples = []
        for seq in self.data:
            # 1. 过滤越界token
            if ((seq < 0) | (seq >= 30000)).any():
                continue

            # 2. 过滤全PAD样本
            if (seq == self.pad_token_id).all():
                continue

            # 3. 过滤无意义短序列（全一样的token）
            if len(set(seq)) == 1:
                continue

            valid_samples.append(seq)

        self.data = np.array(valid_samples)
        self.length = len(self.data)

        print(f"加载数据集：{data_path}，样本数：{self.length}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        Args:
            idx: 样本索引
        Returns:
            input_ids: 输入token id，shape [max_seq_len-1]
            labels: 标签token id，shape [max_seq_len-1]
        """
        # 获取完整序列
        full_seq = self.data[idx]

        # 自回归样本构建：输入是前n-1个token，标签是后n-1个token
        input_ids = full_seq[:-1]
        labels = full_seq[1:]
        # 转换为torch tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return input_ids, labels

class PretrainDataLoader:
    """
    预训练数据加载器封装
    提供训练/验证数据加载器
    """
    def __init__(self,
                 train_path: str,
                 val_path: str,
                 pad_token_id: int,
                 batch_size: int = 8,
                 num_workers: int = 0,
    ):
        """
        Args:
            train_path: 训练集路径
            val_path: 验证集路径
            pad_token_id: PAD token id
            batch_size: 批次大小
            num_workers: 数据加载进程数
        """
        self.train_path = train_path
        self.val_path = val_path
        self.pad_token_id = pad_token_id
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器"""
        train_dataset = PretrainDataset(self.train_path, self.pad_token_id)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return train_loader

    def get_val_loader(self) -> DataLoader:
        """获取验证数据加载器"""
        val_dataset = PretrainDataset(self.val_path, self.pad_token_id)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return val_loader