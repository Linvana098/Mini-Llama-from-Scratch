import torch
import torch.nn as nn
from typing import Optional

class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数
    替代传统ReLU，结合Swish和GLU，提升模型表达能力
    公式：Swish(x) = x * sigmoid(beta*x) → SwiGLU = Swish(x1) * x2
    """
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None):
        """
        Args:
            in_features: 输入维度（hidden_size）
            hidden_features: 中间层维度（通常是in_features*4）
            out_features: 输出维度（默认等于in_features）
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        # 线性层1：in_features -> hidden_features*2（后面拆分成门控和主体）
        self.w1 = nn.Linear(in_features, hidden_features * 2, bias=False)
        # 线性层2：hidden_features -> out_features
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        # Swish的beta参数（设为1，与Llama一致）
        self.beta = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入，shape [batch_size, seq_len, hidden_size]
        Returns:
            激活后的输出，shape [batch_size, seq_len, out_features]
        """
        # 1. 线性变换：[batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_features*2]
        x = self.w1(x)
        # 2. 拆分为两部分：门控和主体，各占一半维度
        x1, x2 = x.chunk(2, dim=-1)
        # 3. Swish激活：x1 * sigmoid(beta*x1)
        # [batch_size, seq_len, hidden_features * 2] -> [batch_size, seq_len, hidden_features]
        swish = x1 * torch.sigmoid(self.beta * x1)
        # 4. GLU: [batch_size, seq_len, hidden_features]
        swiglu = swish * x2
        # 5. 输出线性变换 [batch_size, seq_len, hidden_features] -> [batch_size, seq_len, hidden_size]
        output = self.w2(swiglu)
        return output

    def extra_repr(self) -> str:
        """打印模型参数时的额外信息"""
        return f"in_features={self.in_features}, hidden_features={self.w1.out_features // 2}, out_features={self.w2.out_features}"