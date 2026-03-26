#!/bin/bash
# 文本生成一键脚本
# 功能：加载训练好的模型，根据提示词生成文本
# 使用方式：bash scripts/generate_text.sh "提示词" [可选参数]
# 示例：bash scripts/generate_text.sh "人工智能的未来是" --max_new_tokens 200 --temperature 0.7

# 引入通用工具函数
source scripts/utils.sh

# ====================== 解析命令行参数 ======================
# 检查提示词是否传入
#if [ $# -eq 0 ]; then
#    log_error "请传入生成提示词！示例：bash scripts/generate_text.sh \"人工智能的未来是\""
#fi

# 基础参数
#PROMPT="$1"
#shift  # 移除第一个参数（提示词）

# 默认生成参数
MAX_NEW_TOKENS=20
PROMPT="诗曰"
TEMPERATURE=0.7
TOP_K=50
TOP_P=0.9
STOP_TOKEN=""
USE_CPU=0
BEST_MODEL=1  # 1使用best_model.pth，0使用final_model.pth

# 解析可选参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --stop_token)
            STOP_TOKEN="$2"
            shift 2
            ;;
        --cpu)
            USE_CPU=1
            shift 1
            ;;
        --use_final_model)
            BEST_MODEL=0
            shift 1
            ;;
        *)
            log_warn "未知参数：$1，将忽略"
            shift 1
            ;;
    esac
done

# ====================== 配置参数 ======================
PROJECT_ROOT=$(cd $(dirname $0)/..; pwd)
log_info "项目根目录：$PROJECT_ROOT"

# 配置文件和模型路径
MODEL_CONFIG="$PROJECT_ROOT/configs/model_config.yaml"
DATA_CONFIG="$PROJECT_ROOT/configs/data_config.yaml"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
if [ $BEST_MODEL -eq 1 ]; then
    MODEL_PATH="$CHECKPOINT_DIR/best_model.pth"
else
    MODEL_PATH="$CHECKPOINT_DIR/final_model.pth"
fi
TOKENIZER_DIR=$(get_config_value $DATA_CONFIG "data_paths.tokenizer_dir")

# ====================== 前置检查 ======================
# 环境检查
check_python_env
check_dependencies
if [ $USE_CPU -eq 0 ]; then
    check_gpu
fi

# 文件检查
check_file_exists $MODEL_CONFIG
check_file_exists $DATA_CONFIG
check_file_exists $MODEL_PATH
check_dir_exists $TOKENIZER_DIR

# ====================== 核心生成逻辑 ======================
log_info "========== 开始文本生成 =========="
log_info "生成参数："
log_info "  提示词：$PROMPT"
log_info "  最大生成token数：$MAX_NEW_TOKENS"
log_info "  温度：$TEMPERATURE"
log_info "  Top-K：$TOP_K"
log_info "  Top-P：$TOP_P"
log_info "  模型路径：$MODEL_PATH"

python3 - << EOF
import sys
sys.path.append("$PROJECT_ROOT")
import yaml
import torch
from src.model.decoder_model import DecoderOnlyModel
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.generate.text_generator import TextGenerator

# ====================== 加载配置和组件 ======================
# 模型配置
with open("$MODEL_CONFIG", 'r') as f:
    model_config = yaml.safe_load(f)

# 数据配置
with open("$DATA_CONFIG", 'r') as f:
    data_config = yaml.safe_load(f)

# 设备选择
device = "cuda" if torch.cuda.is_available() and $USE_CPU == 0 else "cpu"
print(f"使用设备：{device}")

# 加载分词器
tokenizer = BPETokenizer(data_config["tokenizer"])
tokenizer.load()

# 加载模型
model = DecoderOnlyModel(model_config)

# 加载权重
checkpoint = torch.load("$MODEL_PATH", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"成功加载模型权重：$MODEL_PATH")

# 初始化生成器
generator = TextGenerator(model, tokenizer, device=device)

# ====================== 生成文本 ======================
print("\n========== 生成结果 ==========")
#print(f"提示词：{$PROMPT}")
print("-" * 50)

# 生成文本
generated_text = generator.generate(
    prompt="$PROMPT",
    max_gen_tokens=$MAX_NEW_TOKENS,
    temperature=$TEMPERATURE,
    top_k=$TOP_K,
    top_p=$TOP_P,
    stop_token="$STOP_TOKEN"
)

# 打印结果
print(generated_text)
print("-" * 50)

# 保存生成结果（可选）
with open("$PROJECT_ROOT/results/sample_generations.txt", "a", encoding="utf-8") as f:
    f.write(f"【提示词】：$PROMPT\n")
    f.write(f"【生成参数】：max_new_tokens={$MAX_NEW_TOKENS}, temperature={$TEMPERATURE}, top_k={$TOP_K}, top_p={$TOP_P}\n")
    f.write(f"【生成结果】：{generated_text}\n")
    f.write("=" * 80 + "\n\n")

print(f"\n生成结果已保存至：$PROJECT_ROOT/results/sample_generations.txt")
EOF

# ====================== 完成 ======================
log_info "========== 文本生成完成 =========="