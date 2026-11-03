"""
Qwen2.5 ChatML 对话模板与多轮消息拼装。

模板格式：
  <|im_start|>system
  你是一个有用的助手。<|im_end|>
  <|im_start|>user
  你好<|im_end|>
  <|im_start|>assistant
"""

from __future__ import annotations

from .tokenizer import ChatTokenizer

DEFAULT_SYSTEM_PROMPT = "你是 Star，一个运行在用户自己电脑上的本地大模型助手，基于 Qwen2.5 架构。你的用户叫 Stay。请用简洁准确的中文回答问题，语气自然友好，并始终记得用户是 Stay。"


def render_chatml(messages: list[dict], system_prompt: str | None = DEFAULT_SYSTEM_PROMPT) -> str:
    """把 openai 风格的 messages 渲染成 ChatML 文本。"""
    parts: list[str] = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    # 追加 assistant 起始标记，引导模型开始回答
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def build_prompt_ids(tokenizer: ChatTokenizer, messages: list[dict],
                     system_prompt: str | None = DEFAULT_SYSTEM_PROMPT) -> list[int]:
    text = render_chatml(messages, system_prompt)
    return tokenizer.encode(text, add_special=False)
