"""
命令行对话程序：加载本地 Qwen2.5 权重，在终端里多轮聊天，流式打印。

用法：
  python scripts/chat_cli.py --model models/Qwen2.5-1.5B-Instruct
输入 exit 或 quit 退出，输入 clear 清空对话历史。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from mini_llm.chat import DEFAULT_SYSTEM_PROMPT, build_prompt_ids
from mini_llm.generation import GenConfig, generate_stream
from mini_llm.tokenizer import ChatTokenizer
from mini_llm.weights import build_model_from_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = build_model_from_dir(args.model, device=device, dtype=dtype)
    tokenizer = ChatTokenizer(args.model)
    gen_cfg = GenConfig(max_new_tokens=args.max_tokens,
                        temperature=args.temperature, top_p=args.top_p)

    print(f"模型已加载（{device}），参数量 {model.num_parameters() / 1e9:.2f}B")
    history: list[dict] = []

    while True:
        try:
            user = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("exit", "quit"):
            break
        if user.lower() == "clear":
            history.clear()
            print("（对话历史已清空）")
            continue
        if not user:
            continue

        history.append({"role": "user", "content": user})
        prompt_ids = build_prompt_ids(tokenizer, history, args.system)

        print("AI > ", end="", flush=True)
        answer_ids: list[int] = []
        for tid, piece in generate_stream(model, tokenizer, prompt_ids, gen_cfg, device):
            if tid == "__done__":
                answer_ids = piece
                break
            print(piece, end="", flush=True)
        print()
        history.append({"role": "assistant",
                        "content": tokenizer.decode(answer_ids)})


if __name__ == "__main__":
    main()
