import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .decoder_layer import DecoderLayer
from .rms_norm import RMSNorm
import math

class DecoderOnlyModel(nn.Module):
    """
    完整的Decoder-only模型（Llama/Qwen风格）
    结构：Embedding → 多层DecoderLayer → RMSNorm → 输出投影
    """
    def __init__(self, config: dict):
        """
        Args:
            config: 模型配置（来自model_config.yaml）
                - vocab_size: 词汇表大小
                - hidden_dim: 模型维度
                - num_heads: 注意力头数
                - num_layers: Decoder层数
                - max_seq_len: 最大序列长度
                - intermediate_size: SwiGLU中间层维度
                - dropout: dropout率
                - bias: 是否使用偏置
        """
        super(DecoderOnlyModel, self).__init__()
        self.config = config['model']
        self.vocab_size = self.config['vocab_size']
        self.hidden_dim = self.config['hidden_dim']
        self.num_heads = self.config['num_heads']
        self.num_layers = self.config['num_layers']
        self.max_seq_len = self.config['max_seq_len']
        # self.intermediate_size = config['intermediate_size']
        self.dropout = self.config['dropout']
        self.bias = self.config['bias']

        # 1. embedding层
        self.embeddings = nn.Embedding(self.vocab_size, self.hidden_dim)

        # 2. Dropout层
        self.embedding_dropout = nn.Dropout(self.dropout)

        # 3. Decoder逐层堆叠
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                max_position_embeddings=self.max_seq_len,
                dropout=self.dropout,
                bias=self.bias,
            )
            for _ in range(self.num_layers)
        ])

        # 4. 最终归一化层
        self.norm = RMSNorm(self.hidden_dim)

        # 5. 输出投影层
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        # 权重绑定：embedding层和输出投影层共享权重（Llama/Qwen风格）
        self.output_proj.weight = self.embeddings.weight

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        前向传播
        Args:
            input_ids: 输入token id，shape [batch_size, seq_len]
            attention_mask: 注意力掩码，shape [batch_size, seq_len]
            return_attn_weights: 是否返回各层注意力权重
        Returns:
            logits: 输出logits，shape [batch_size, seq_len, vocab_size]
            attn_weights: 各层注意力权重（可选）
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Token Embedding
        x = self.embeddings(input_ids)  # [batch_size, seq_len, hidden_dim]
        x = self.embedding_dropout(x)

        # 2. 初始化注意力权重
        all_attn_weights = [] if return_attn_weights else None

        # 3. 逐层进入Decoder layer
        for layer in self.layers:
            x, attn_weights = layer(x, attention_mask)
            if return_attn_weights:
                all_attn_weights.append(attn_weights)

        # 4. 最终归一化
        x = self.norm(x)

        # 5. 输出投影
        logits = self.output_proj(x)   # [batch, seq_len, vocab_size]

        return logits, all_attn_weights

    @torch.no_grad()
    def generate_token(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        生成单个token（自回归生成核心）
        采样方法：Top-K + Top-P 混合采样
        Args:
            input_ids: 输入token id，shape [batch_size, seq_len]
            **kwargs:
                - temperature: 温度系数（控制随机性，默认1.0）
                - top_k: Top-K采样数（默认50，设为0则禁用）
                - top_p: Top-P采样阈值（默认0.9，设为1.0则禁用）
        Returns:
            生成的token id，shape [batch_size, 1]
        """
        # 前向传播获取logits
        logits, _ = self.forward(input_ids)    # [batch_size, seq_len, vocab_size]

        # 取最后一个token
        next_token_logits = logits[:, -1, :]    # [batch_size, vocab_size]

        # 温度调节
        temperature = kwargs.get('temperature', 1.0)
        if temperature > 0 and temperature != 1.0:
            next_token_logits /= temperature
        elif temperature == 0:
            # 温度为0就是贪心
            return next_token_logits.argmax(dim=-1, keepdim=True)

        # Top-K + Top-P 混合采样
        top_k = kwargs.get('top_k', 50)
        top_p = kwargs.get('top_p', 0.9)

        # 1. 应用Top-K筛选
        if top_k > 0:
            # 获取Top-K的logits值
            top_k_values, top_k_indices = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
            # 创建掩码：只保留Top-K的token，其余设为-∞
            next_token_logits = torch.scatter(
                torch.full_like(next_token_logits, float("-inf")),
                dim=-1,
                index=top_k_indices,
                src=top_k_values,
            )

        # 2. 再应用Top-P筛选
        if 0 < top_p < 1.0:
            # 对logits做softmax得到概率
            probs = torch.softmax(next_token_logits, dim=-1)
            # 排序概率（降序）并计算累计概率
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # 筛选出累计概率 <= top_p 的token, 保留第一个超过top_p的token，避免空集
            cutoff_mask = cumulative_probs <= top_p
            cutoff_mask[:, 1:] = cutoff_mask[:, :-1].clone()
            cutoff_mask[:, 0] = True

            # 按排序后的索引，重新排列原始logits，应用掩码（取反cutoff_mask）
            sorted_logits = torch.gather(next_token_logits, dim=-1, index=sorted_indices)
            sorted_logits[~cutoff_mask] = float("-inf")

            # 还原回原始索引空间
            next_token_logits = torch.scatter(
                torch.full_like(next_token_logits, float("-inf")),
                dim=-1,
                index=sorted_indices,
                src=sorted_logits,
            )

        # 3. 计算最终概率分布
        probs = torch.softmax(next_token_logits, dim=-1)

        # 4. 采样下一个token
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token

