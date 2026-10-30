"""
分词器封装：使用底层 tokenizers 库直接加载 tokenizer.json，
不经过 transformers.AutoTokenizer，并补充 Qwen 特殊 token 常量。
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer


class ChatTokenizer:
    PAD = "<|endoftext|>"
    EOS = "<|im_end|>"
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.eos_id = self.tok.token_to_id(self.EOS)
        self.pad_id = self.tok.token_to_id(self.PAD)
        self.im_start_id = self.tok.token_to_id(self.IM_START)
        self.im_end_id = self.tok.token_to_id(self.IM_END)

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        return self.tok.encode(text, add_special_tokens=add_special).ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special)

    def decode_single(self, token_id: int) -> str:
        # 单 token 解码，用于流式输出
        return self.tok.decode([token_id], skip_special_tokens=False)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()
