"""
从零训练器：在自己的语料上训练（或继续预训练）手写 Transformer。

包含：数据批次构造、交叉熵、AdamW、warmup + cosine 学习率调度、
梯度裁剪、梯度累积、safetensors 检查点保存。
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .model import MiniLLM, ModelConfig
from .tokenizer import ChatTokenizer


# --------------------------------------------------------------------------- #
# 数据：把整段文本按固定长度切成训练块，下一个 token 即预测目标
# --------------------------------------------------------------------------- #
class TextDataset:
    def __init__(self, tokenizer: ChatTokenizer, text_path: str | Path, seq_len: int):
        text = Path(text_path).read_text(encoding="utf-8")
        ids = tokenizer.encode(text, add_special=False)
        self.seq_len = seq_len
        # 每个块长度 seq_len+1，输入取前 n 个、标签错一位
        self.chunks = [
            ids[i:i + seq_len + 1]
            for i in range(0, len(ids) - seq_len - 1, seq_len)
        ]
        if not self.chunks:
            raise ValueError("文本太短，无法切出训练块，请增加语料或减小 seq_len")

    def __len__(self):
        return len(self.chunks)

    def get_batch(self, indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for idx in indices:
            chunk = self.chunks[idx]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return (torch.tensor(xs, device=device), torch.tensor(ys, device=device))


def cosine_lr(step: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    if step >= total:
        return min_lr
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


class Trainer:
    def __init__(self, model: MiniLLM, tokenizer: ChatTokenizer,
                 lr: float = 3e-4, weight_decay: float = 0.1,
                 warmup_ratio: float = 0.03, grad_clip: float = 1.0,
                 device: str = "auto"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.grad_clip = grad_clip
        self.warmup_ratio = warmup_ratio

        # 权重衰减只作用于矩阵参数；bias、归一化一维参数不衰减
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        self.optim = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=False)

    def train(self, dataset: TextDataset, epochs: int, batch_size: int,
              grad_accum: int = 1, save_dir: str | Path | None = None,
              log_every: int = 10):
        steps_per_epoch = max(1, len(dataset) // batch_size)
        total_steps = steps_per_epoch * epochs // grad_accum
        warmup = max(1, int(total_steps * self.warmup_ratio))
        base_lr = self.optim.param_groups[0]["lr"]

        step = 0
        self.model.train()
        for epoch in range(epochs):
            perm = torch.randperm(len(dataset)).tolist()
            for batch_start in range(0, len(perm), batch_size):
                indices = perm[batch_start:batch_start + batch_size]
                x, y = dataset.get_batch(indices, self.device)
                logits, _ = self.model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                (loss / grad_accum).backward()

                if (batch_start // batch_size + 1) % grad_accum == 0:
                    lr = cosine_lr(step, warmup, total_steps, base_lr, base_lr * 0.1)
                    for g in self.optim.param_groups:
                        g["lr"] = lr
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optim.step()
                    self.optim.zero_grad()
                    step += 1
                    if step % log_every == 0:
                        print(f"epoch {epoch} step {step}/{total_steps} "
                              f"loss {loss.item():.4f} lr {lr:.2e}")

        if save_dir:
            self.save_checkpoint(save_dir)

    def save_checkpoint(self, save_dir: str | Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # 去掉 "model." 之类的前缀差异，直接按模块名保存
        tensors = {k: v.contiguous().cpu() for k, v in self.model.state_dict().items()}
        save_file(tensors, str(save_dir / "model.safetensors"))
        print(f"检查点已保存到 {save_dir}")
