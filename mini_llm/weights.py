"""
权重加载器：直接读取 HuggingFace / ModelScope 格式的 safetensors 权重，
把官方 Qwen2.5 参数名映射到我们手写的模块，不借助 transformers。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .model import MiniLLM, ModelConfig


def _load_index(model_dir: Path) -> dict:
    """读取 model.safetensors.index.json（权重分片时）或直接枚举单文件。"""
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return json.load(f)["weight_map"]
    single = model_dir / "model.safetensors"
    if single.exists():
        return {}
    raise FileNotFoundError(f"在 {model_dir} 下找不到 safetensors 权重文件")


def load_weights(model: MiniLLM, model_dir: str | Path,
                 dtype: torch.dtype = torch.float16) -> MiniLLM:
    model_dir = Path(model_dir)
    cfg = model.cfg
    weight_map = _load_index(model_dir)

    # 收集所有分片文件
    if weight_map:
        files = sorted(set(weight_map.values()))
    else:
        files = ["model.safetensors"]

    # 我们自己的 state_dict key
    own_state = model.state_dict()
    loaded_keys: set[str] = set()

    for fname in files:
        with safe_open(model_dir / fname, framework="pt") as st:
            for official_key in st.keys():
                # 官方命名带 "model." 前缀，lm_head 单独处理
                if official_key == "lm_head.weight":
                    if cfg.tie_word_embeddings:
                        continue  # tied 权重无需单独加载
                    local_key = "lm_head.weight"
                elif official_key.startswith("model."):
                    local_key = official_key[len("model."):]
                else:
                    local_key = official_key

                if local_key not in own_state:
                    continue

                tensor = st.get_tensor(official_key)
                # 归一化层保持 float32 更稳定，其余按目标 dtype
                if "norm" in local_key:
                    tensor = tensor.to(torch.float32)
                else:
                    tensor = tensor.to(dtype)
                own_state[local_key].copy_(tensor)
                loaded_keys.add(local_key)

    # tied embedding 自检：确认 lm_head 未被错误要求
    missing = set(own_state.keys()) - loaded_keys
    expected_missing = set()
    if cfg.tie_word_embeddings:
        expected_missing.add("lm_head.weight") if "lm_head.weight" in own_state else None
    unexpected_missing = missing - expected_missing
    if unexpected_missing:
        raise RuntimeError(f"以下权重未能从检查点加载: {sorted(unexpected_missing)}")

    return model


def build_model_from_dir(model_dir: str | Path,
                         device: str | torch.device = "auto",
                         dtype: torch.dtype = torch.float16) -> MiniLLM:
    """一站式：读配置 -> 建模型 -> 加载权重 -> 放到设备。"""
    model_dir = Path(model_dir)
    cfg = ModelConfig.from_json(model_dir / "config.json")
    model = MiniLLM(cfg)
    load_weights(model, model_dir, dtype=dtype)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model
