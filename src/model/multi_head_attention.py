import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .rope import RoPE

class MultiHeadAttention(nn.Module):

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
            bias: 是否使用偏置（Llama/Qwen均禁用）
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 校验：hidden_dim 必须能被 num_heads 整除
        assert self.head_dim * num_heads == self.hidden_dim, "hidden_dim must be divisible by num_heads"

        # qkv线性投影层：
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # 输出投影层
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # RoPE位置编码
        self.rope = RoPE(self.head_dim, max_position_embeddings)
        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 缓存缩放因子
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # 缩放因子注册为模型缓冲区
        self.register_buffer("scale_dk", self.scale)

    def _prepare_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成因果掩码（Causal Mask）
        逻辑：上三角矩阵为-∞，下三角为0，确保每个token只能看到前面的token
        Args:
            seq_len: 序列长度
            device: 设备
        Returns:
            掩码矩阵，shape [1, 1, seq_len, seq_len]
        """
        # 生成下三角矩阵（True表示可见，False表示不可见）
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
        # 转换为注意力掩码（False -> -inf，True -> 0）
        mask = mask.masked_fill(mask == False, float("-inf")).masked_fill(mask == True, float(0.0))
        # [seq_len, seq_len] → [1, 1, seq_len, seq_len]
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        Args:
            x: 输入，shape [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码，shape [batch_size, seq_len]
        Returns:
            output: 注意力输出，shape [batch_size, seq_len, hidden_dim]
            attn_weights: 注意力权重（可选），shape [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # 1. qkv投影
        # x: [batch_size, seq_len, hidden_dim] -> qkv: [batch_size, seq_len, hidden_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 拆分为多头格式
        # [batch_size, seq_len, hidden_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 应用RoPE编码到q和k
        q = self.rope(q)
        k = self.rope(k)

        # 4. 计算注意力分数
        # 公式：score = q @ k.T / sqrt(head_dim)
        # q: [batch_size, num_heads, seq_len, head_dim]
        # k.T: [batch_size, num_heads, head_dim, seq_len]
        # attn_scores: [batch_size, num_heads, seq_len, seq_len]
        attn_scores = q @ k.transpose(-2, -1) / self.scale_dk.to(device)

        # 5. 应用因果掩码
        causal_mask = self._prepare_causal_mask(seq_len, device=device)
        attn_scores += causal_mask

        # 6. 应用padding掩码（如果有）
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len] → [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores += attention_mask

        # 7. 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        # 应用dropout
        attn_weights = self.dropout(attn_weights)

        # 8. 加权求和
        # attn_scores: [batch_size, num_heads, seq_len, seq_len]
        # v: [batch_size, num_heads, seq_len, head_dim]
        # output: [batch_size, num_heads, seq_len, head_dim]
        output = attn_weights @ v

        # 9. 重塑回x原始形状
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 10.输出投影
        output = self.out_proj(output)

        return output, attn_weights
