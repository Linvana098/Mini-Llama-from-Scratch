import torch
import torch.nn as nn
from typing import List, Optional, Union
from src.model.decoder_model import DecoderOnlyModel
from src.tokenizer.bpe_tokenizer import BPETokenizer

class TextGenerator:
    """
    文本生成器
    基于自回归逻辑，实现文本生成
    支持temperature、top-p/top_k混合生成参数
    """
    def __init__(self, model: DecoderOnlyModel, tokenizer: BPETokenizer, device: str = 'cuda'):
        """
        Args:
            model: 训练好的Decoder-only模型
            tokenizer: BPE分词器
            device: 生成设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else torch.device('cpu'))

        # 移动模型到设备并设为评估模式
        self.model.eval()
        self.model.to(self.device)

        print(f"文本生成器初始化完成，设备：{self.device}")

    @torch.no_grad()
    def generate(self,
                 prompt: str,
                 max_gen_tokens: int = 100,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 stop_token: Optional[str] = None,
    ) -> str:
        """
        生成文本
        Args:
            prompt: 提示词
            max_gen_tokens: 最大生成token数
            temperature: 温度（控制生成多样性，0-1之间）
            top_p: Top-P采样阈值（累计概率阈值，0.8-0.95最佳）
            top_k: Top-K采样数（候选token数）
            stop_token: 停止生成的token
        Returns:
            生成的完整文本
        """
        # 1. 编码提示词
        input_ids = self.tokenizer.encode(prompt)   # [seq_len]

        input_ids = torch.tensor(input_ids.ids, dtype=torch.long).unsqueeze(0).to(self.device)  # [1, seq_len]

        # 2. 自回归生成
        for i in range(max_gen_tokens):
            # 生成单个token  [1, 1]
            next_token = self.model.generate_token(
                input_ids,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            # 把新生成的token添加到输入
            input_ids = torch.cat([input_ids, next_token], dim=-1)  # [1, seq_len]

            # 检查停止条件
            next_token_str = self.tokenizer.decode(next_token.squeeze(0).cpu().numpy())
            if stop_token is not None and stop_token != "" and stop_token in next_token_str:
                break
            if input_ids.shape[1] == max_gen_tokens:
                break

        # 3. 解码生成的token
        generated_ids = input_ids.squeeze().cpu().numpy()
        generated_text = self.tokenizer.decode(generated_ids)

        return generated_text


