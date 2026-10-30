"""
冒烟测试：无需下载大模型，用随机初始化的迷你模型验证核心逻辑。

重点验证：
  1. 前向传播输出形状正确
  2. KV Cache 正确性——增量解码的 logits 必须与一次性 prefill 一致
  3. 采样器输出合法 token
  4. 生成器能正常结束

运行：python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from mini_llm.model import MiniLLM, ModelConfig
from mini_llm.generation import GenConfig, _sample


def tiny_config():
    return ModelConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )


def test_forward_shape():
    model = MiniLLM(tiny_config()).eval()
    ids = torch.randint(0, 1000, (2, 16))
    logits, caches = model(ids)
    assert logits.shape == (2, 16, 1000), logits.shape
    assert len(caches) == 2
    print("[PASS] 前向传播形状正确")


def test_kv_cache_consistency():
    """增量路径与一次性路径的 logits 必须逐位置一致。"""
    cfg = tiny_config()
    model = MiniLLM(cfg).eval()
    ids = torch.randint(0, 1000, (1, 8))

    # 一次性 prefill
    with torch.no_grad():
        full_logits, _ = model(ids)

    # 逐 token 增量喂入
    with torch.no_grad():
        caches = None
        inc_logits = []
        for t in range(8):
            cur = ids[:, t:t + 1]
            pos = torch.tensor([[t]])
            logits, caches = model(cur, position_ids=pos, kv_caches=caches)
            inc_logits.append(logits)
        inc_logits = torch.cat(inc_logits, dim=1)

    diff = (full_logits - inc_logits).abs().max().item()
    assert diff < 1e-4, f"KV Cache 不一致，最大误差 {diff}"
    print(f"[PASS] KV Cache 增量解码与 prefill 一致（最大误差 {diff:.2e}）")


def test_sampler():
    logits = torch.randn(3, 1000)
    cfg = GenConfig(temperature=0.7, top_k=20, top_p=0.9)
    tok = _sample(logits, cfg)
    assert 0 <= tok < 1000
    greedy = _sample(logits, GenConfig(temperature=0))
    assert greedy == logits[-1].argmax().item()
    print("[PASS] 采样器与贪心解码正常")


def test_generate_loop():
    cfg = tiny_config()
    model = MiniLLM(cfg).eval()

    # 构造一个最小假 tokenizer，仅需 eos_id / decode_single
    class FakeTok:
        eos_id = 999
        def decode_single(self, i): return "x"

    prompt = [1, 2, 3, 4]
    count = 0
    with torch.no_grad():
        for tid, _ in __import__("mini_llm.generation", fromlist=["generate_stream"]).generate_stream(
                model, FakeTok(), prompt, GenConfig(max_new_tokens=10, temperature=0), "cpu"):
            if tid == "__done__":
                break
            count += 1
    assert count <= 10
    print(f"[PASS] 生成循环正常（生成 {count} 个 token 后正常结束）")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_forward_shape()
    test_kv_cache_consistency()
    test_sampler()
    test_generate_loop()
    print("\n全部冒烟测试通过 ✓")
