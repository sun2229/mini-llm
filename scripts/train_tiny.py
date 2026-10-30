"""
从零训练示例：用一个迷你 Transformer 在自己的语料上训练语言模型。

用法：
  python scripts/train_tiny.py --data data/sample.txt --out checkpoints/tiny --seq_len 256

没有 GPU 也能跑，模型尺寸刻意调小以便快速验证训练流程。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from mini_llm.model import MiniLLM, ModelConfig
from mini_llm.tokenizer import ChatTokenizer
from mini_llm.train import TextDataset, Trainer


def build_tiny_config(vocab_size: int) -> ModelConfig:
    # 8 层、6 个注意力头、2 个 KV 头的迷你模型，约几千万参数
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_size=384,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=6,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample.txt")
    parser.add_argument("--tokenizer", required=True,
                        help="已下载的 Qwen 模型目录，用于提供分词器")
    parser.add_argument("--out", default="checkpoints/tiny")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    tokenizer = ChatTokenizer(args.tokenizer)
    cfg = build_tiny_config(tokenizer.vocab_size)
    model = MiniLLM(cfg)
    print(f"迷你模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    dataset = TextDataset(tokenizer, args.data, args.seq_len)
    print(f"训练块数量: {len(dataset)}")

    trainer = Trainer(model, tokenizer, lr=args.lr)
    trainer.train(dataset, epochs=args.epochs, batch_size=args.batch_size,
                  save_dir=args.out)


if __name__ == "__main__":
    main()
