import json
import os
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from typing import List, Optional, Union
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class BPETokenizer:
    """
    自定义 Byte-Level BPE分词器
    适配 Llama/Qwen分词逻辑，支持训练、编码、解码
    """
    def __init__(self, config: dict):
        """
        初始化分词器
        Args:
            config: 分词器配置（来自data_config.yaml）
                - vocab_size: 词汇表大小
                - min_frequency: 最小词频
                - special_tokens: 特殊token列表
                - tokenizer_dir: 分词器保存路径
        """
        self.config = config
        self.tokenizer = None
        self.special_tokens = config["special_tokens"]
        self.tokenizer_dir = config["tokenizer_dir"]
        self._vocab_size = config["vocab_size"]
        self.min_frequency = config["min_frequency"]

        # 特殊token映射
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        # 创建分词器保存目录
        os.makedirs(self.tokenizer_dir, exist_ok=True)

    def train(self, corpus_path: str):
        """
        训练BPE分词器
        Args:
            corpus_path: 原始语料路径
        """
        # 初始化BPE tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token, fuse_unk=True))
        # normalization（统一全角,半角字符）
        self.tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        # 设置预分词器（ByteLevel）
        self.tokenizer.pre_tokenizer = ByteLevel()
        # 初始化解码器
        self.tokenizer.decoder = ByteLevelDecoder()
        # 初始化训练器
        trainer = BpeTrainer(
            vocab_size=self._vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
        )
        # 读取原始语料
        if os.path.isfile(corpus_path):
            files = [corpus_path]   # 单个文件
        else:
            # 目录和子目录下的所有txt文件
            files = []
            for dirpath, _, filenames in os.walk(corpus_path):
                for filename in filenames:
                    if filename.lower().endswith("_utf8.txt"):
                        file_path = os.path.join(dirpath, filename)
                        files.append(file_path)

        print(f"开始训练分词器，语料文件数：{len(files)}")

        self.tokenizer.train(files=files, trainer=trainer)

        # 设置后处理器：添加<BOS>和<EOS>
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            pair=f"{self.bos_token} $A {self.eos_token} $B:1 {self.eos_token}:1",
            special_tokens=[
                (self.bos_token, self.tokenizer.token_to_id(self.bos_token)),
                (self.eos_token, self.tokenizer.token_to_id(self.eos_token)),
            ],
        )

        # 保存分词器
        self.save()
        print(f"分词器训练完成，保存至：{self.tokenizer_dir}")

    def load(self):
        """加载已训练的分词器"""
        tokenizer_path = os.path.join(self.tokenizer_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            print(f"已从该路径下成功加载分词器：{tokenizer_path}")
        else:
            raise FileNotFoundError(f"该路径下的分词器文件不存在：{tokenizer_path}")

    def save(self):
        """保存分词器到指定目录"""
        tokenizer_path = os.path.join(self.tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        # 额外保存词表
        vocab = self.tokenizer.get_vocab()
        vocab_path = os.path.join(self.tokenizer_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    def encode(self, text: Union[str, List[str]], max_length: Optional[int] = None) -> List[int]:
        """
        文本编码为token id
        Args:
            text: 输入文本（单条或列表）
            max_length: 最大序列长度，超过截断
        Returns:
            token id列表
        """
        if self.tokenizer is None:
            self.load()

        # 批量编码和单条编码
        if isinstance(text, list):
            encodings = self.tokenizer.encode_batch(text)
            ids = [enc.ids for enc in encodings]
        else:
            encoding = self.tokenizer.encode(text)
            ids = encoding.ids

        # 如果超过最大序列长度，截断
        if max_length is not None:
            if isinstance(ids[0], list):
                ids = [i[:max_length] for i in ids]
            else:
                ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int]) -> str:
        """token id解码为文本"""
        if self.tokenizer is None:
            self.load()
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def pad_token_id(self) -> int:
        """获取pad token id"""
        return self.tokenizer.token_to_id(self.pad_token)
    @property
    def bos_token_id(self) -> int:
        """获取bos token id"""
        return self.tokenizer.token_to_id(self.bos_token)
    @property
    def eos_token_id(self) -> int:
        """获取eos token id"""
        return self.tokenizer.token_to_id(self.eos_token)
    @property
    def unk_token_id(self) -> int:
        """获取unk token id"""
        return self.tokenizer.token_to_id(self.unk_token)
    @property
    def vocab_size(self) -> int:
        """获取词汇表大小"""
        return self.tokenizer.get_vocab_size()
