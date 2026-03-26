#!/bin/bash
# 数据预处理一键脚本
# 功能：1. 训练BPE分词器 2. 清洗语料 3. 编码为token id 4. 划分训练/验证集
# 使用方式：bash scripts/prepare_data.sh

# 引入通用工具函数
source scripts/utils.sh

# ====================== 配置参数 ======================
# 基础路径（根据项目根目录调整，若脚本在项目根目录运行则无需修改）
PROJECT_ROOT=$(cd $(dirname $0)/..; pwd)
log_info "项目根目录：$PROJECT_ROOT"

# 配置文件路径
DATA_CONFIG="$PROJECT_ROOT/configs/data_config.yaml"
MODEL_CONFIG="$PROJECT_ROOT/configs/model_config.yaml"

# 语料路径（从配置文件读取）
RAW_CORPUS=$(get_config_value $DATA_CONFIG "data_paths.raw_corpus")
TOKENIZER_DIR=$(get_config_value $DATA_CONFIG "data_paths.tokenizer_dir")

# 预处理参数（从配置文件读取）
VOCAB_SIZE=$(get_config_value $DATA_CONFIG "tokenizer.vocab_size")
MAX_SEQ_LEN=$(get_config_value $DATA_CONFIG "preprocess.max_seq_len")

# ====================== 前置检查 ======================
# 检查基础环境
check_python_env
check_dependencies

# 检查配置文件和原始语料
check_file_exists $DATA_CONFIG
check_file_exists $MODEL_CONFIG
check_dir_exists $RAW_CORPUS

# 检查目录
check_dir_exists $TOKENIZER_DIR
check_dir_exists "$PROJECT_ROOT/data/processed"

# ====================== 核心逻辑 ======================
log_info "========== 开始数据预处理 =========="

# Step 1: 训练BPE分词器
log_info "Step 1/2: 训练BPE分词器（词汇表大小：$VOCAB_SIZE）"
python3 - << EOF
import sys
sys.path.append("$PROJECT_ROOT")
from src.tokenizer.bpe_tokenizer import BPETokenizer
import yaml

# 加载数据配置
with open("$DATA_CONFIG", 'r') as f:
    data_config = yaml.safe_load(f)

# 初始化分词器
tokenizer = BPETokenizer(data_config["tokenizer"])

# 训练分词器
tokenizer.train("$RAW_CORPUS")
EOF
log_info "分词器训练完成，保存至：$TOKENIZER_DIR"

# Step 2: 语料预处理（清洗+编码+划分）
log_info "Step 2/2: 语料预处理（最大序列长度：$MAX_SEQ_LEN）"
python3 - << EOF
import sys
sys.path.append("$PROJECT_ROOT")
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.process_data.data_preprocess import preprocess_corpus
import yaml

# 加载配置
with open("$DATA_CONFIG", 'r') as f:
    data_config = yaml.safe_load(f)

# 加载分词器
tokenizer = BPETokenizer(data_config["tokenizer"])
tokenizer.load()

# 预处理语料
preprocess_corpus(
    corpus_path=data_config["data_paths"]["raw_corpus"],
    tokenizer=tokenizer,
    config=data_config["preprocess"]
)
EOF

# ====================== 完成 ======================
log_info "========== 数据预处理完成 =========="
log_info "✅ 分词器路径：$TOKENIZER_DIR"
log_info "✅ 训练集路径：$(get_config_value $DATA_CONFIG "data_paths.processed_train")"
log_info "✅ 验证集路径：$(get_config_value $DATA_CONFIG "data_paths.processed_val")"