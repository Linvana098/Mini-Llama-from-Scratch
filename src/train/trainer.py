import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple
from tokenizers import Tokenizer
from src.model.decoder_model import DecoderOnlyModel
from src.process_data.data_loader import PretrainDataLoader
from src.generate.text_generator import TextGenerator
from src.tokenizer.bpe_tokenizer import BPETokenizer
import json
import time
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np

class PretrainTrainer():
    """
    预训练训练器
    实现完整的训练循环：前向/反向传播、日志记录、模型保存、验证
    """
    def __init__(self,
                 model: DecoderOnlyModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 config: dict,
                 pad_token_id: int,
                 device: str = 'cuda',
                 ):
        """
        Args:
            model: 模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            config: 训练配置
            pad_token_id: PAD token id
            device: 训练设备
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.pad_token_id = pad_token_id

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 移动模型到gpu
        self.model.to(self.device)
        # 混合精度训练
        # 根据配置文件选择数据类型
        self.dtype = torch.float16 if config["dtype"] == "float16" else torch.float32
        # 初始化梯度缩放器（只在使用float16时启用）
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=(self.dtype == torch.float16))

        # 训练参数
        self.epochs = config["training"]["epochs"]
        self.gradient_clip_norm = config["training"]["gradient_clip_norm"]
        self.log_interval = config["training"]["log_interval"]
        self.eval_interval = config["training"]["eval_interval"]
        self.save_interval = config["training"]["save_interval"]

        # 模型保存配置
        self.patience = config["training"]["patience"]
        self.delta = config["training"]["delta"]
        self.val_loss = 500
        self.val_ppl = 500
        self.early_stop = False
        self.counter = 0
        self.best_loss = np.inf
        self.save_dir = config["checkpoint"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

        # 训练状态
        self.best_val_ppl = float("inf")
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        self.val_ppls = []

        # 临时设置一个生成器，为了训练中测试生成效果
        tokenizer_path = "./data/tokenizer/tokenizer.json"
        tokenizer = Tokenizer.from_file(tokenizer_path)
        self.generator = TextGenerator(self.model, tokenizer, self.device)

        print(f"训练器初始化完成：")
        print(f"  设备：{self.device}")
        print(f"  精度：{self.dtype}")
        print(f"  训练轮次：{self.epochs}")
        print(f"  总训练步数：{len(self.train_loader) * self.epochs}")

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算交叉熵损失（忽略PAD token）
        Args:
            logits: 模型输出logits，shape [batch_size, seq_len, vocab_size]
            labels: 标签，shape [batch_size, seq_len]
        Returns:
            平均损失值
        """

        # 转换logits到float32（只在使用float16时需要）
        if logits.dtype == torch.float16:
            logits = logits.float()

        # 保证两个张量在同一设备
        labels = labels.to(logits.device)

        # 计算交叉熵（忽略<pad>）
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),         # [batch_size * seq_len, vocab_size]
            labels.reshape(-1),                     # [batch_size * seq_len]
            ignore_index=-self.pad_token_id,
            reduction="mean",
        )

        return loss

    def compute_ppl(self, loss: torch.Tensor) -> float:
        """
        计算困惑度（Perplexity）
        PPL = exp(loss)，越低表示模型效果越好
        """
        # 限制最大loss值，溢出保护
        # detach: 脱离计算图，不再追踪梯度
        loss_clipped = torch.clamp(loss.detach(), max=20.0)
        return torch.exp(loss_clipped).item()

    def trian_one_epoch(self, epoch, log_path, writer):
        """训练单个epoch"""
        # 模型设定为训练模式
        self.model.train()
        strat_time = time.time()
        epoch_loss = 0

        for batch_idx, (input_ids, labels) in enumerate(self.train_loader):
            # 移动数据到设备
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # 1. 前向传播（混合精度）
            with torch.amp.autocast(device_type='cuda', enabled=(self.dtype == torch.float16)):
                # 模型前向
                logits, _ = self.model(input_ids)
                # 计算损失
                loss = self.compute_loss(logits, labels)

            # 2. 反向传播
            self.optimizer.zero_grad()
            # 如果使用了float16，先缩放损失
            self.grad_scaler.scale(loss).backward()

            # 3. 梯度裁剪
            if self.gradient_clip_norm > 0.0:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            # 4. 更新权重
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # 5. 更新学习率
            self.scheduler.step()

            # 6. 记录损失
            epoch_loss += loss.item()
            self.train_losses.append(loss.item())
            self.global_step += 1

            # 7. 打印日志
            if self.global_step % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                ppl = self.compute_ppl(loss)
                elapsed = time.time() - strat_time
                print(
                    f"Epoch: {epoch+1}/{self.epochs} | "
                    f"Step: {self.global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"PPL: {ppl:.2f} | "
                    f"LR: {lr:.6f} | "
                    f"Time: {elapsed:.2f}s"
                )

            # 8. 验证（早停）
            if self.global_step % self.eval_interval == 0:
                self.val_loss, self.val_ppl = self.validate()
                self.val_losses.append(self.val_loss)
                self.val_ppls.append(self.val_ppl)

                print(f"Validation | Step: {self.global_step} | Loss: {self.val_loss:.4f} | PPL: {self.val_ppl:.2f}")

                if self.val_loss < self.best_loss - self.delta:
                    # 验证集变好 → 保存模型
                    self.best_loss = self.val_loss
                    self.best_val_ppl = self.val_ppl
                    self.counter = 0
                    self.save_model(f"best_model.pth")
                    print(f"保存最优模型，PPL: {self.best_val_ppl:.2f}")
                else:
                    # 验证集没变好 → 计数+1
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True

                # 回到训练模式
                self.model.train()

                # 尝试生成
                self.model.eval()
                sample_text1 = self.gen_sample(prompt="人工智能启动自检程序，", max_new_tokens=32)
                sample_text2 = self.gen_sample(prompt="当人类第一次接触到外星智慧，", max_new_tokens=32)
                writer.add_text("Generated_Text/Sample", sample_text1, self.global_step)
                writer.add_text("Generated_Text/Sample", sample_text2, self.global_step)
                print(sample_text1)
                print(sample_text2)
                # 回到训练模式
                self.model.train()

            # 9. 保存日志
            if self.global_step % 100 == 0:
                # 训练中，每 500 个 step 执行一次
                log_entry = {
                    "step": self.global_step,
                    "epoch": epoch + 1,
                    "train_loss": round(loss.item(), 4),
                    "val_loss": round(self.val_loss, 4),
                    "train_ppl": round(ppl, 4),
                    "val_ppl": round(self.val_ppl, 4),
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                # 读取现有日志 → 添加新记录 → 保存
                with open(log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)

                log_data["logs"].append(log_entry)

                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)

                # tensorboard
                # Loss
                writer.add_scalar("Loss/train", loss.item(), self.global_step)
                writer.add_scalar("Loss/val", self.val_loss, self.global_step)

                # PPL
                writer.add_scalar("Metrics/train_PPL", ppl, self.global_step)
                writer.add_scalar("Metrics/val_PPL", self.val_ppl, self.global_step)

                # 学习率
                writer.add_scalar("Optimizer/lr", lr, self.global_step)

            if self.early_stop:
                break

        # 计算epoch平均损失
        avg_epoch_loss = epoch_loss / len(self.train_loader)
        avg_epoch_ppl = self.compute_ppl(torch.tensor(avg_epoch_loss))

        print(f"Epoch {epoch + 1} 完成 | 平均Loss: {avg_epoch_loss:.4f} | 平均PPL: {avg_epoch_ppl:.2f}")

        # 保存epoch模型
        if (epoch + 1) % self.save_interval == 0:
            self.save_model(f"model_epoch_{epoch + 1}.pth")

    @torch.no_grad()
    def gen_sample(self, prompt: str, max_new_tokens: int) -> str:
        return self.generator.generate(
                    prompt=prompt,
                    max_gen_tokens=max_new_tokens,
                    temperature =0.5,
                    top_k =50,
                    top_p =0.9,
        )

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        val_loss = 0

        for input_ids, labels in self.val_loader:
            # 移动数据到设备
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            with torch.amp.autocast(device_type='cuda', enabled=(self.dtype == torch.float16)):
                logits, _ = self.model(input_ids)
                loss = self.compute_loss(logits, labels)

            val_loss += loss.item()

        # 计算平均损失和PPL
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_ppl = self.compute_ppl(torch.tensor(avg_val_loss))

        return avg_val_loss, avg_val_ppl

    def save_model(self, filename: str):
        """保存模型权重"""
        save_path = os.path.join(self.save_dir, filename)

        # 保存模型状态
        torch.save({
            "epoch": self.global_step // len(self.train_loader),
            "step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_ppl": self.best_val_ppl,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_ppls": self.val_ppls,
        }, save_path)

    def train(self):

        # 自动创建日志目录
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=f"runs/llm_run_{current_time}")

        # 日志保存路径
        log_path = "./logs/train_logs.json"
        # 初始化日志结构
        log_data = {
            "model_name": "mini-llama",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "logs": []
        }
        # 保存初始日志
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        """完整训练流程"""
        print("="*50)
        print("开始预训练")
        print("="*50)

        start_time = time.time()

        for epoch in range(self.epochs):
            if not self.early_stop:
                self.trian_one_epoch(epoch, log_path, writer)
            else:
                print(f"触发早停机制，当前 Step: {self.global_step}")
                break

        # 训练结束
        total_time = time.time() - start_time
        print("="*50)
        print(f"训练完成！总耗时：{total_time/3600:.2f}小时")
        print(f"最优验证集PPL：{self.best_val_ppl:.2f}")
        print("="*50)

        # 保存最终模型
        self.save_model("final_model.pth")

        # 返回训练结果
        return {
            "best_val_ppl": self.best_val_ppl,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_ppls": self.val_ppls,
            "total_steps": self.global_step
        }