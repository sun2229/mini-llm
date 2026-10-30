"""
从零手搓的现代 Decoder-Only Transformer，架构对齐 Qwen2.5。

本文件不使用 transformers 的任何模型类，所有核心组件均手写：
  - RMSNorm            : 均方根层归一化
  - RoPE               : 旋转位置编码
  - GQA                : 分组查询注意力（含 KV Cache）
  - SwiGLU             : SiLU 门控前馈网络
  - tied embedding     : 输入输出词表权重共享
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    vocab_size: int = 151936
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        # Qwen2.5 中 head_dim = hidden_size / attention_heads
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_groups(self) -> int:
        # 每个 KV 头需要被多少个 Q 头共享
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 只取我们需要的字段，忽略 tokenizer 等无关键
        fields = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in fields})


# --------------------------------------------------------------------------- #
# RMSNorm：用 1/RMS(x) 做归一化，再乘可学习缩放，比 LayerNorm 少一次均值计算
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 转 float32 计算均方根，保证数值稳定，再转回原 dtype
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# --------------------------------------------------------------------------- #
# RoPE：把位置信息以“旋转”的方式编码进 Q/K，让注意力感知相对位置
# --------------------------------------------------------------------------- #
def precompute_rope(head_dim: int, max_seq_len: int, theta: float,
                    device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    # 频率序列：theta^(-2i/d)，i 取 0..d/2-1
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    # [seq_len, head_dim/2]
    freqs = torch.outer(positions, inv_freq)
    # 拼成 d 维：前半与后半相同频率
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: [batch, heads, seq, head_dim]；cos/sin: [seq, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        # 旋转公式：前半与后半配对，x_rot = [-x_{d/2:}, x_{:d/2}]
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

    return rotate(q), rotate(k)


# --------------------------------------------------------------------------- #
# GQA：多组 Q 头共享同一组 K/V 头，减少 KV 缓存体积
# --------------------------------------------------------------------------- #
def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    b, kv_heads, seq, d = x.shape
    # 展开后 reshape 成与 Q 头数一致
    x = x[:, :, None, :, :].expand(b, kv_heads, groups, seq, d)
    return x.reshape(b, kv_heads * groups, seq, d)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size
        q_heads = cfg.num_attention_heads
        kv_heads = cfg.num_key_value_heads

        # Q 投影到全部查询头；K/V 只投影到少量 KV 头
        self.q_proj = nn.Linear(h, q_heads * cfg.head_dim, bias=True)
        self.k_proj = nn.Linear(h, kv_heads * cfg.head_dim, bias=True)
        self.v_proj = nn.Linear(h, kv_heads * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(q_heads * cfg.head_dim, h, bias=True)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: tuple[torch.Tensor, torch.Tensor] | None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, seq, _ = x.shape
        h, kv_h, d = self.cfg.num_attention_heads, self.cfg.num_key_value_heads, self.cfg.head_dim

        q = self.q_proj(x).view(b, seq, h, d).transpose(1, 2)
        k = self.k_proj(x).view(b, seq, kv_h, d).transpose(1, 2)
        v = self.v_proj(x).view(b, seq, kv_h, d).transpose(1, 2)

        # RoPE 必须在缓存拼接之前作用于当前 token
        q, k = apply_rope(q, k, cos, sin)

        # 拼接历史 KV，实现增量解码
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        k = repeat_kv(k, self.cfg.kv_groups)
        v = repeat_kv(v, self.cfg.kv_groups)

        # 因果掩码交给 is_causal：prefill 阶段遮挡未来 token
        is_prefill = q.shape[2] > 1 and k.shape[2] == q.shape[2]
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_prefill,
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, seq, h * d)
        return self.o_proj(out), new_cache


# --------------------------------------------------------------------------- #
# SwiGLU 前馈：down( silu(gate(x)) * up(x) )，门控让网络学得更平滑
# --------------------------------------------------------------------------- #
class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# 单层 Transformer Block（Pre-Norm 结构）
# --------------------------------------------------------------------------- #
class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = FeedForward(cfg)

    def forward(self, x, cos, sin, kv_cache):
        residual = x
        h, new_cache = self.self_attn(self.input_layernorm(x), cos, sin, kv_cache)
        x = residual + h

        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache


# --------------------------------------------------------------------------- #
# 完整模型
# --------------------------------------------------------------------------- #
class MiniLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # tie 时 lm_head 不单独建参数，推理时直接复用 embedding 权重
        self.lm_head = None if cfg.tie_word_embeddings else nn.Linear(
            cfg.hidden_size, cfg.vocab_size, bias=False)

        # RoPE 缓存延迟到第一次 forward 时按设备建立
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def _ensure_rope(self, device: torch.device, dtype: torch.dtype, seq_len: int):
        if self._cos is None or self._cos.device != device or len(self._cos) < seq_len:
            self._cos, self._sin = precompute_rope(
                self.cfg.head_dim,
                max(self.cfg.max_position_embeddings, seq_len),
                self.cfg.rope_theta, device, dtype,
            )

    def forward(self, input_ids: torch.Tensor,
                position_ids: torch.Tensor | None = None,
                kv_caches: list | None = None) -> tuple[torch.Tensor, list]:
        b, seq = input_ids.shape
        device = input_ids.device
        self._ensure_rope(device, self.embed_tokens.weight.dtype,
                          seq if position_ids is None else int(position_ids.max()) + 1)

        if position_ids is None:
            position_ids = torch.arange(seq, device=device).unsqueeze(0).expand(b, -1)
        positions = position_ids[0]  # batch 内位置相同
        cos = self._cos[positions]
        sin = self._sin[positions]

        x = self.embed_tokens(input_ids)

        new_caches: list = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, cache = layer(x, cos, sin, cache)
            new_caches.append(cache)

        x = self.norm(x)
        head_weight = self.embed_tokens.weight if self.cfg.tie_word_embeddings else self.lm_head.weight
        logits = F.linear(x, head_weight)
        return logits, new_caches

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset_caches(self) -> list:
        """生成一轮新对话时调用，返回与层数等长的空缓存占位。"""
        return [None] * self.cfg.num_hidden_layers
