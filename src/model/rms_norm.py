import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMSNorm 归一化层
    替代LayerNorm，计算效率更高，训练更稳定
    Llama/Qwen均采用RMSNorm
    """
    def __init__(self, hidden_dim: int, eps: float = 1e-9):
        """
        Args:
            hidden_dim: 输入维度
            eps: 防止除零的小值
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        # 可学习的缩放参数 [hidden_dim]
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """核心归一化逻辑"""
        # RMSNorm公式：x / RMS(x) * weight
        # RMS(x) = sqrt(mean(x^2) + eps)
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入，shape [batch_size, seq_len, hidden_dim]
        Returns:
            归一化后的输出，shape与输入一致
        """
        # 先转化为float32计算均方根，再转回原始精度 [batch_size, seq_len, hidden_dim]
        output = self._norm(x.float()).type_as(x)
        # 应用缩放参数 [batch_size, seq_len, hidden_dim]
        return output * self.weight

    def extra_repr(self) -> str:
        """打印模型参数时的额外信息"""
        return f"hidden_dim={self.weight.shape[0]}, eps={self.eps}"