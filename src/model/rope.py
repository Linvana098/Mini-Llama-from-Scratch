import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) 位置编码
    核心逻辑：将位置信息融入token embedding，实现长序列泛化
    适配Llama/Qwen的RoPE实现
    """
    def __init__(self, head_dim: int, max_position_embeddings: int = 8192):
        """
        Args:
            head_dim: 每个注意力头的维度（hidden_size / num_heads）
            max_position_embeddings: 支持的最大序列长度
        """
        super(RoPE, self).__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings

        # 预计算频率矩阵（theta）
        # 公式：theta_i = 10000^(-2(i-1)/head_dim)，i从1到head_dim/2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))   # [head_dim / 2]
        self.register_buffer("inv_freq", inv_freq)  # [head_dim / 2]

        # 预计算位置索引（0到max_position_embeddings-1）
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings),
            persistent=False
        )   # [max_position_embeddings]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：对token embedding应用RoPE编码
        Args:
            x: 输入token embedding，分好头之后的，shape [batch_size, num_heads, seq_len, head_dim]
        Returns:
            编码后的embedding，shape与输入一致
        """
        batch_size, _, seq_len, head_dim = x.shape

        # 1. 处理位置索引
        position_ids = self.position_ids[:seq_len]

        # 2. 计算频率（position * inv_freq） [seq_len, head_dim / 2]
        freqs = torch.outer(position_ids, self.inv_freq)

        # 3. 计算cos/sin [seq_len, head_dim / 2]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        # [batch_size,seq_len, head_dim / 2]
        cos = cos.unsqueeze(0).repeat(batch_size, 1, 1)
        sin = sin.unsqueeze(0).repeat(batch_size, 1, 1)
        # [batch_size, seq_len, head_dim / 2] -> [batch_size, 1, seq_len, head_dim / 2]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # 4. 将x按维度拆分（用于旋转） [batch_size, num_heads, seq_len, head_dim // 2]
        x1 = x[..., : self.head_dim // 2]   # 前半
        x2 = x[..., self.head_dim // 2 :]   # 后半

        # 5. 计算旋转后的embedding
        # 公式：
        # x1_rot = x1 * cos(emb) - x2 * sin(emb)
        # x2_rot = x1 * sin(emb) + x2 * cos(emb)
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        # 6. 拼接
        x_rot =torch.cat([x1_rot, x2_rot], dim=-1)

        return x_rot

    @staticmethod
    def apply_rope_to_attention(
            q: torch.Tensor,
            k: torch.Tensor,
            head_dim: int,
            max_position_embeddings: int = 8192,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对注意力的query和key应用RoPE（静态方法）
        Args:
            q: query tensor，shape [batch_size, num_heads, seq_len, head_dim]
            k: key tensor，shape [batch_size, num_heads, seq_len, head_dim]
            head_dim: 每个头的维度
            max_position_embeddings: 最大序列长度
        Returns:
            应用RoPE后的q和k
        """
        rope = RoPE(head_dim, max_position_embeddings)
        q_rot = rope(q)
        k_rot = rope(k)
        return q_rot, k_rot




