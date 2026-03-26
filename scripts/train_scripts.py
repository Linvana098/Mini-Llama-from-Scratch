import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
from src.model.decoder_model import DecoderOnlyModel
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.train.optim_scheduler import get_optimizer, get_lr_scheduler
from src.train.trainer import PretrainTrainer
from src.process_data.data_preprocess import preprocess_corpus
from src.process_data.data_loader import PretrainDataLoader
import yaml

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
file_path = "./configs/train_config.yaml"
try:
    with open(file_path, "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)
        print("✅ 数据处理配置文件读取成功！")
except:
    print(f"❌ 错误：找不到配置文件！路径：{file_path}")

batch_size = train_config["training"]["batch_size"]
epochs = train_config["training"]["epochs"]
lr = train_config["optimizer"]["lr"]
save_dir = train_config["checkpoint"]["save_dir"]

print("========== 开始模型训练 ==========")
print("训练参数：")
print(f"  批次大小：  {batch_size}")
print(f"  训练轮次：  {epochs}")
print(f"  初始学习率： {lr}")
print(f"  保存目录：  {save_dir}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备：{device}")

# ====================== 初始化组件 ======================

# 1. 加载分词器
tokenizer = BPETokenizer(data_config["tokenizer"])
tokenizer.load()
pad_token_id = tokenizer.pad_token_id

# 2. 初始化数据加载器
data_loader = PretrainDataLoader(
    train_path=data_config["data_paths"]["processed_train"],
    val_path=data_config["data_paths"]["processed_val"],
    pad_token_id=pad_token_id,
    batch_size=batch_size,
    num_workers=0
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

# ====================== 完成 ======================
print("========== 训练流程结束 ==========")
print(f"✅ 模型权重保存至：{save_dir}")
print(f"✅ 最优模型：{save_dir}/best_model.pth")
print(f"✅ 最终模型：{save_dir}/final_model.pth")