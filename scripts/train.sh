#!/bin/bash
# 模型训练一键脚本
# 功能：1. 加载配置 2. 初始化模型/优化器/数据加载器 3. 开始训练 4. 保存模型
# 使用方式：bash scripts/train.sh [--cpu]（可选--cpu强制使用CPU）

# 引入通用工具函数
source scripts/utils.sh

# ====================== 解析命令行参数 ======================
USE_CPU=0
if [ "$1" == "--cpu" ]; then
    USE_CPU=1
    log_warn "强制使用CPU训练（仅建议测试）"
fi

# ====================== 配置参数 ======================
PROJECT_ROOT=$(cd $(dirname $0)/..; pwd)
log_info "项目根目录：$PROJECT_ROOT"

# 配置文件路径
MODEL_CONFIG="$PROJECT_ROOT/configs/model_config.yaml"
TRAIN_CONFIG="$PROJECT_ROOT/configs/train_config.yaml"
DATA_CONFIG="$PROJECT_ROOT/configs/data_config.yaml"

# 核心路径
TRAIN_DATA=$(get_config_value $DATA_CONFIG "data_paths.processed_train")
VAL_DATA=$(get_config_value $DATA_CONFIG "data_paths.processed_val")
TOKENIZER_DIR=$(get_config_value $DATA_CONFIG "data_paths.tokenizer_dir")
CHECKPOINT_DIR=$(get_config_value $TRAIN_CONFIG "checkpoint.save_dir")

# 训练参数
BATCH_SIZE=$(get_config_value $TRAIN_CONFIG "training.batch_size")
EPOCHS=$(get_config_value $TRAIN_CONFIG "training.epochs")
LR=$(get_config_value $TRAIN_CONFIG "optimizer.lr")

# ====================== 前置检查 ======================
# 环境检查
check_python_env
check_dependencies
if [ $USE_CPU -eq 0 ]; then
    check_gpu
fi

# 文件检查
check_file_exists $MODEL_CONFIG
check_file_exists $TRAIN_CONFIG
check_file_exists $DATA_CONFIG
check_file_exists $TRAIN_DATA
check_file_exists $VAL_DATA
check_dir_exists $TOKENIZER_DIR
check_dir_exists $CHECKPOINT_DIR

# ====================== 核心训练逻辑 ======================
log_info "========== 开始模型训练 =========="
log_info "训练参数："
log_info "  批次大小：$BATCH_SIZE"
log_info "  训练轮次：$EPOCHS"
log_info "  初始学习率：$LR"
log_info "  模型配置：$MODEL_CONFIG"
log_info "  保存目录：$CHECKPOINT_DIR"

python3 - << EOF

import torch
from src.model.decoder_model import DecoderOnlyModel
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.process_data.data_loader import PretrainDataLoader
from src.train.optim_scheduler import get_optimizer, get_lr_scheduler
from src.train.trainer import PretrainTrainer

# ====================== 加载配置 ======================
# 模型配置
with open("$MODEL_CONFIG", 'r') as f:
    model_config = yaml.safe_load(f)

# 训练配置
with open("$TRAIN_CONFIG", 'r') as f:
    train_config = yaml.safe_load(f)

# 数据配置
with open("$DATA_CONFIG", 'r') as f:
    data_config = yaml.safe_load(f)

# ====================== 设备选择 ======================
device = "cuda" if torch.cuda.is_available() and $USE_CPU == 0 else "cpu"
print(f"使用设备：{device}")

# ====================== 初始化组件 ======================
# 1. 加载分词器
tokenizer = BPETokenizer(data_config["tokenizer"])
tokenizer.load()
pad_token_id = tokenizer.pad_token_id

# 2. 初始化数据加载器
data_loader = PretrainDataLoader(
    train_path="$TRAIN_DATA",
    val_path="$VAL_DATA",
    pad_token_id=pad_token_id,
    batch_size=$BATCH_SIZE,
    num_workers=0  # 新手建议设为0，避免多进程问题
)
train_loader = data_loader.get_train_loader()
val_loader = data_loader.get_val_loader()

# 3. 初始化模型
model = DecoderOnlyModel(model_config)
print(f"模型参数总量：{sum(p.numel() for p in model.parameters()):,}")

# 4. 初始化优化器和学习率调度器
optimizer = get_optimizer(model, train_config)
scheduler = get_lr_scheduler(optimizer, train_config)

# ====================== 开始训练 ======================
trainer = PretrainTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    config=train_config,
    pad_token_id=pad_token_id,
    device=device
)

# 启动训练
train_results = trainer.train()

# 打印训练结果
print("\n========== 训练结果 ==========")
print(f"最优验证集PPL：{train_results['best_val_ppl']:.2f}")
print(f"总训练步数：{train_results['total_steps']}")

EOF

# ====================== 完成 ======================
log_info "========== 训练流程结束 =========="
log_info "✅ 模型权重保存至：$CHECKPOINT_DIR"
log_info "✅ 最优模型：$CHECKPOINT_DIR/best_model.pth"
log_info "✅ 最终模型：$CHECKPOINT_DIR/final_model.pth"
#print(f"训练完成时间：$(__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"))")