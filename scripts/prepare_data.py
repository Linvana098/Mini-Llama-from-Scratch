import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.process_data.data_preprocess import preprocess_corpus
import yaml

print("========== 开始数据预处理 ==========")
print("开始读取配置文件...")

### 读取配置文件
file_path = "./configs/data_config.yaml"
try:
    with open(file_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
        print("✅ 数据处理配置文件读取成功！")
except:
    print(f"❌ 错误：找不到配置文件！路径：{file_path}")

file_path = "./configs/model_config.yaml"
try:
    with open(file_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
        print("✅ 模型配置文件读取成功！")
except:
    print(f"❌ 错误：找不到配置文件！路径：{file_path}")

# Step 1: 训练BPE分词器
vocab_size = data_config["tokenizer"]["vocab_size"]
print(f"Step 1/2: 训练BPE分词器（词汇表大小：{vocab_size}）")

# 初始化分词器
tokenizer = BPETokenizer(data_config["tokenizer"])

# 训练分词器
tokenizer.train(data_config["data_paths"]["raw_corpus"])

print(f"分词器训练完成，保存至：{data_config["tokenizer"]["tokenizer_dir"]}")


# Step 2: 语料预处理（清洗+编码+划分）
max_seq_len = model_config["model"]["max_seq_len"]
print(f"Step 2/2: 语料预处理（最大序列长度：{max_seq_len}）")

# 加载分词器
tokenizer = BPETokenizer(data_config["tokenizer"])
tokenizer.load()

# 预处理语料
preprocess_corpus(
    corpus_path=data_config["data_paths"]["raw_corpus"],
    tokenizer=tokenizer,
    config=data_config["preprocess"]
)

# ====================== 完成 ======================
print("========== 数据预处理完成 ==========")
print(f"✅ 分词器路径：{data_config["tokenizer"]["tokenizer_dir"]}")
print(f"✅ 训练集路径：{data_config["preprocess"]["processed_train"]}")
print(f"✅ 验证集路径：{data_config["preprocess"]["processed_val"]}")