"""
推理生成器：手写采样策略与 KV Cache 增量解码。

支持：
  - temperature / top-k / top-p(nucleus) 采样
  - repetition_penalty 重复惩罚
  - 贪心解码（temperature=0）
  - 逐 token 流式生成
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import MiniLLM
from .tokenizer import ChatTokenizer


@dataclass
class GenConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1


def _sample(logits: torch.Tensor, cfg: GenConfig) -> int:
    """从最后一个位置的 logits 按策略采样一个 token。"""
    logits = logits.float()

    # 重复惩罚：已出现过的 token 按概率反向缩放
    if cfg.repetition_penalty != 1.0:
        # 注意：这里的惩罚在 generate 循环中配合历史 token 使用
        pass

    if cfg.temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / cfg.temperature

    # top-k：只保留概率最高的 k 个
    if cfg.top_k > 0:
        k = min(cfg.top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k)
        logits[logits < topk_vals[-1]] = float("-inf")

    # top-p：在 top-k 基础上累计概率截断
    probs = F.softmax(logits, dim=-1)
    if cfg.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        # 第一个超过阈值的 token 之后全部移除
        mask = cum_probs > cfg.top_p
        mask[1:] = mask[:-1].clone()
        mask[0] = False
        remove = sorted_idx[mask]
        probs[remove] = 0.0
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, num_samples=1).item())


def _apply_repetition_penalty(logits: torch.Tensor, input_ids: list[int],
                              penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    unique = set(input_ids)
    for tid in unique:
        if logits[tid] > 0:
            logits[tid] /= penalty
        else:
            logits[tid] *= penalty
    return logits


@torch.no_grad()
def generate_stream(model: MiniLLM, tokenizer: ChatTokenizer, prompt_ids: list[int],
                    cfg: GenConfig, device: str | torch.device):
    """
    流式生成器：每次 yield 一个新 token 的 id 与解码文本。
    结束时 yield 一个特殊标记 ("__done__", output_ids)。
    """
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    history = list(prompt_ids)

    # ---- Prefill：一次性喂入整个 prompt，建立各层 KV Cache ----
    logits, kv_caches = model(input_ids)
    next_id = _sample(_apply_repetition_penalty(logits[0, -1], history, cfg.repetition_penalty), cfg)
    history.append(next_id)

    output_ids: list[int] = []
    for step in range(cfg.max_new_tokens):
        if next_id == tokenizer.eos_id:
            break
        output_ids.append(next_id)
        piece = tokenizer.decode_single(next_id)
        yield next_id, piece

        # ---- Decode：只喂一个新 token，靠缓存避免重复计算 ----
        cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
        pos = torch.tensor([[len(history) - 1]], dtype=torch.long, device=device)
        logits, kv_caches = model(cur, position_ids=pos, kv_caches=kv_caches)
        next_id = _sample(
            _apply_repetition_penalty(logits[0, -1], history, cfg.repetition_penalty), cfg)
        history.append(next_id)

    yield "__done__", output_ids


@torch.no_grad()
def generate(model: MiniLLM, tokenizer: ChatTokenizer, prompt_ids: list[int],
             cfg: GenConfig, device: str | torch.device) -> tuple[str, list[int]]:
    """非流式接口：内部复用流式逻辑，返回完整文本。"""
    output_ids: list[int] = []
    for tid, piece in generate_stream(model, tokenizer, prompt_ids, cfg, device):
        if tid == "__done__":
            output_ids = piece
    return tokenizer.decode(output_ids), output_ids
