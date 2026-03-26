import os
import re
import numpy as np
from typing import Tuple
from src.tokenizer.bpe_tokenizer import BPETokenizer

def clean_text(text: str, min_length: int = 50) -> str:
    """
    清洗单条文本，去除低质量内容
    Args:
        text: 原始文本
        min_length: 最小文本长度（字符数）
    Returns:
        清洗后的文本，空字符串表示低质量
    """
    # 1. 去除特殊字符和乱码
    text = re.sub(r'[^\u4e00-\u9fff_a-zA-Z0-9\s,.!?;:()"''-]', '', text)
    text = re.sub("（本书资料收集于网上，版权归原作者所有）", '', text)
    text = re.sub("本书资料收集于网上，版权归原作者所有", '', text)
    text = re.sub("Xinty665 免费制作", '', text)
    text = re.sub("说明：本书借用【云中孤雁】制作的模板", '', text)
    # 2. 去除多余空格和换行
    text = re.sub(r'\s+', ' ', text).strip()
    # 3. 过滤过短的文本
    if len(text) < min_length:
        return ""
    return text

def preprocess_corpus(
        corpus_path: str,
        tokenizer: BPETokenizer,
        config: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    完整的语料预处理流程
    Args:
        corpus_path: 原始语料路径
        tokenizer: 已训练的BPE分词器
        config: 数据配置（来自data_config.yaml）
            - max_seq_len: 最大序列长度
            - min_text_len: 最小文本长度
            - val_split: 验证集占比
    Returns:
        train_ids: 训练集token id，shape [num_train_samples, max_seq_len]
        val_ids: 验证集token id，shape [num_val_samples, max_seq_len]
    """
    max_seq_len = config["max_seq_len"]
    min_text_len = config["min_text_len"]
    val_split = config["val_split"]

    # 1. 读取原始语料
    print(f"开始读取语料：{corpus_path}")
    files = []
    for dirpath, _, filenames in os.walk(corpus_path):
        for filename in filenames:
            if filename.lower().endswith("_utf8.txt"):
                file_path = os.path.join(dirpath, filename)
                files.append(file_path)
    clean_paragraphs = []
    for file in files:
        with open(file, "r", encoding=config["encoding"]) as f:
            raw_text = f.read()
        # 2. 按换行拆分
        paragraphs = raw_text.split("\n")
        # 3. 清洗文本
        for para in paragraphs:
            clean_para = clean_text(para, min_text_len)
            if clean_para:
                clean_paragraphs.append(clean_para)

    print(f"清洗后有效段落数：{len(clean_paragraphs)}")
    # 4. 编码
    print("开始编码文本...")
    all_ids = []
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size
    assert pad_id < vocab_size, f"pad_id={pad_id} 超过词表大小={vocab_size}"

    for para in clean_paragraphs:
        ids = tokenizer.encode(para)
        # 滑窗，max_seq_len
        for i in range(0, len(ids) - max_seq_len, max_seq_len):
            chunk = ids[i:i + max_seq_len]
            # 长度不足max_seq_len的用<pad>补充
            if len(chunk) < max_seq_len:
                chunk += [pad_id] * (max_seq_len - len(chunk))
            all_ids.append(chunk)
    # 转换为numpy数组
    all_ids = np.array(all_ids, dtype=np.int64)
    print(f"编码完成，总样本数：{len(all_ids)}")
    # 5. 划分训练集和验证集
    val_size = int(len(all_ids) * val_split)
    # 打乱数据
    np.random.shuffle(all_ids)
    train_ids = all_ids[:-val_size]
    val_ids = all_ids[-val_size:]
    print(f"训练集样本数：{len(train_ids)}, 验证集样本数：{len(val_ids)}")
    # 6. 保存处理后的数据
    np.save(config["processed_train"], train_ids)
    np.save(config["processed_val"], val_ids)
    print(f"预处理完成，数据保存至：")
    print(f"  训练集：{config['processed_train']}")
    print(f"  验证集：{config['processed_val']}")

    return train_ids, val_ids