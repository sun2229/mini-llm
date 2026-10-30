"""
FastAPI 后端：加载手写模型，提供 REST 与 SSE 流式聊天接口。

接口：
  GET  /                  聊天页面
  GET  /api/health        模型与设备信息
  POST /api/chat          非流式对话
  POST /api/chat/stream   SSE 流式对话
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mini_llm.chat import DEFAULT_SYSTEM_PROMPT, build_prompt_ids
from mini_llm.generation import GenConfig, generate, generate_stream
from mini_llm.tokenizer import ChatTokenizer
from mini_llm.weights import build_model_from_dir

# 项目根目录与默认模型路径（可通过环境变量覆盖）
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__import__("os").environ.get("MINI_LLM_MODEL", ROOT / "models" / "Qwen2.5-1.5B-Instruct"))

app = FastAPI(title="Mini-LLM", version="1.0.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


# --------------------------------------------------------------------------- #
# 模型单例：启动时加载一次
# --------------------------------------------------------------------------- #
class Engine:
    model = None
    tokenizer = None
    device = None
    dtype = None

    @classmethod
    def load(cls):
        if cls.model is not None:
            return
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.dtype = torch.float16 if cls.device == "cuda" else torch.float32
        cls.model = build_model_from_dir(MODEL_DIR, device=cls.device, dtype=cls.dtype)
        cls.tokenizer = ChatTokenizer(MODEL_DIR)


@app.on_event("startup")
def startup():
    if MODEL_DIR.exists():
        Engine.load()
    else:
        print(f"[警告] 模型目录不存在: {MODEL_DIR}，请先运行 download_model.py")


# --------------------------------------------------------------------------- #
# 请求结构
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 512
    repetition_penalty: float = 1.1
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/health")
def health():
    if Engine.model is None:
        return {"status": "no_model", "model_dir": str(MODEL_DIR)}
    return {
        "status": "ok",
        "device": Engine.device,
        "model": str(MODEL_DIR.name),
        "parameters_billion": round(Engine.model.num_parameters() / 1e9, 2),
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    Engine.load()
    messages = [m.model_dump() for m in req.messages]
    prompt_ids = build_prompt_ids(Engine.tokenizer, messages, req.system_prompt)
    cfg = GenConfig(max_new_tokens=req.max_tokens, temperature=req.temperature,
                    top_p=req.top_p, top_k=req.top_k,
                    repetition_penalty=req.repetition_penalty)
    text, out_ids = generate(Engine.model, Engine.tokenizer, prompt_ids, cfg, Engine.device)
    return {"role": "assistant", "content": text, "tokens": len(out_ids)}


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    Engine.load()
    messages = [m.model_dump() for m in req.messages]
    prompt_ids = build_prompt_ids(Engine.tokenizer, messages, req.system_prompt)
    cfg = GenConfig(max_new_tokens=req.max_tokens, temperature=req.temperature,
                    top_p=req.top_p, top_k=req.top_k,
                    repetition_penalty=req.repetition_penalty)

    def event_gen():
        for tid, piece in generate_stream(Engine.model, Engine.tokenizer,
                                          prompt_ids, cfg, Engine.device):
            if tid == "__done__":
                yield f"data: {json.dumps({'done': True, 'tokens': len(piece)}, ensure_ascii=False)}\n\n"
                break
            yield f"data: {json.dumps({'delta': piece}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
