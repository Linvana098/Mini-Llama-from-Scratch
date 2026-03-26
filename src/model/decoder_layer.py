import torch
import torch.nn as nn
from typing import Optional, Tuple
from .multi_head_attention import MultiHeadAttention
from .rms_norm import RMSNorm
from .swi_glu import SwiGLU

class DecoderLayer(nn.Module):
    """
    单层Decoder结构
    核心架构：MHA → RMSNorm → 残差 → SwiGLU → RMSNorm → 残差
    Llama/Qwen的Decoder层设计
    """
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 max_position_embeddings: int = 8192,
                 dropout: float = 0.1,
                 bias: bool = False,
    ):
        """
        Args:
            hidden_dim: 模型维度
            num_heads: 注意力头数
            max_position_embeddings: RoPE最大序列长度
            dropout: dropout率
            bias: 是否使用偏置
        """
        super(DecoderLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_dim = hidden_dim * 4

        # 1. 注意力层前的归一化层
        self.input_norm = RMSNorm(hidden_dim=hidden_dim)
        # 2. 多头注意力层
        self.self_attn = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            bias=bias,
        )
        # 3. 前馈层后的归一化层
        self.post_attn_norm = RMSNorm(hidden_dim=hidden_dim)
        # 4. SwiGLU前馈层
        self.swi_glu = SwiGLU(
            in_features=hidden_dim,
            hidden_features=self.intermediate_dim,
            out_features=hidden_dim,
        )
        # 5. Dropout层
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        Args:
            x: 输入，shape [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码
        Returns:
            output: Decoder层输出，shape与输入一致
            attn_weights: 注意力权重（可选）
        """

        # 1. 注意力层(Pre-Norm)
        x_norm = self.input_norm(x)
        attn_output, attn_weights = self.self_attn(x_norm, attention_mask=attention_mask)
        attn_output = self.dropout(attn_output)

        # 2. 残差连接
        x = x + attn_output

        # 3. SwiGLU前馈层(Pre-Norm)
        x_norm = self.post_attn_norm(x)
        ffn_output = self.swi_glu(x_norm)
        ffn_output = self.dropout(ffn_output)

        # 4. 残差连接
        x = x + ffn_output

        return x, attn_weights

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 正态分布初始化，std=0.01 小范围，防止爆炸
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def extra_repr(self) -> str:
        """打印模型参数时的额外信息"""
        return f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, intermediate_dim={self.intermediate_dim}"