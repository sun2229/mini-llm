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
import os
import sys
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mini_llm.chat import DEFAULT_SYSTEM_PROMPT, build_prompt_ids
from mini_llm.generation import GenConfig, generate, generate_stream
from mini_llm import image_gen
from mini_llm.model_manager import ModelManager
from mini_llm.tokenizer import ChatTokenizer
from mini_llm.weights import build_model_from_dir

# 项目根目录与默认模型路径（可通过环境变量覆盖）
# 打包后（PyInstaller frozen）cwd 已切到 exe 目录，models/ 与其同级
if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MINI_LLM_MODELS", ROOT / "models"))
MODEL_DIR = Path(os.environ.get("MINI_LLM_MODEL", "")) or (MODELS_DIR / "Qwen2.5-0.5B-Instruct")
GENERATED_DIR = ROOT / "generated"

app = FastAPI(title="Star", version="1.1.0")
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")
manager = ModelManager(MODELS_DIR)


# --------------------------------------------------------------------------- #
# 模型单例：启动时加载一次；模型缺失时由 /api/chat 触发动态加载
# --------------------------------------------------------------------------- #
class Engine:
    model = None
    tokenizer = None
    device = None
    dtype = None
    model_dir: Path | None = None

    @classmethod
    def load(cls) -> bool:
        if cls.model is not None:
            return True
        target = cls._find_model_dir()
        if target is None or not (target / "config.json").exists():
            return False
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.dtype = torch.float16 if cls.device == "cuda" else torch.float32
        try:
            cls.model = build_model_from_dir(target, device=cls.device, dtype=cls.dtype)
        except Exception as e:  # noqa: BLE001
            print(f"[错误] 模型加载失败: {e}")
            return False
        cls.tokenizer = ChatTokenizer(target)
        cls.model_dir = Path(target)
        return True

    @classmethod
    def _find_model_dir(cls) -> Path | None:
        if not MODELS_DIR.exists():
            return None
        dirs = sorted(d for d in MODELS_DIR.iterdir()
                      if d.is_dir() and (d / "config.json").exists())
        return dirs[0] if dirs else None


@app.on_event("startup")
def startup():
    if Engine.load():
        print(f"[OK] 模型已加载: {Engine.model_dir}")
    else:
        print(f"[提示] 模型未就绪，前端将引导下载（目录: {MODELS_DIR}）")


# --------------------------------------------------------------------------- #
# 模型管理接口（Star 桌面版用）
# --------------------------------------------------------------------------- #
@app.get("/api/model-status")
def model_status():
    models = manager.list_models()
    return {
        "has_model": Engine.model is not None or len(models) > 0,
        "models": models,
        "loading": Engine.model is not None,
        "gpu": manager.has_nvidia_gpu(),
        "torch_cuda": torch.cuda.is_available(),
    }


@app.post("/api/download-model")
def download_model():
    state = manager.download_default()
    return {"started": state["running"], "state": state}


@app.get("/api/download-progress")
def download_progress():
    return manager.download_progress()


# --------------------------------------------------------------------------- #
# 图像生成接口（Star 本地绘画）
# --------------------------------------------------------------------------- #
class GenImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 4
    width: int = 512
    height: int = 512
    seed: int | None = None


@app.get("/api/image-status")
def image_status():
    return {
        "available": image_gen.sd_available(MODELS_DIR),
        "loaded": image_gen.ImageGenerator.pipe is not None,
        "gpu": manager.has_nvidia_gpu(),
        "torch_cuda": torch.cuda.is_available(),
        "model_dir": str(image_gen.sd_model_dir(MODELS_DIR)),
    }


@app.post("/api/download-sd")
def download_sd():
    state = manager.download_repo(
        image_gen.SD_REPO_ID, image_gen.sd_model_dir(MODELS_DIR))
    return {"started": state["running"], "state": state}


@app.post("/api/gen-image")
def gen_image(req: GenImageRequest):
    if not req.prompt.strip():
        return {"error": "请先输入画面描述"}
    if not image_gen.ImageGenerator.load(MODELS_DIR):
        return {"error": "绘画模型未就绪，请先下载", "download_required": True}
    import time as _t
    t0 = _t.time()
    try:
        img, seed = image_gen.ImageGenerator.generate(
            req.prompt.strip(), req.negative_prompt.strip(),
            steps=max(1, min(req.steps, 12)),
            width=req.width, height=req.height, seed=req.seed)
    except Exception as e:  # noqa: BLE001
        return {"error": f"生成失败: {e}"}
    fname = f"star_{_t.strftime('%Y%m%d_%H%M%S')}_{seed}.png"
    out = GENERATED_DIR / fname
    img.save(out, format="PNG")
    return {
        "url": f"/generated/{fname}",
        "seed": seed,
        "elapsed_s": round(_t.time() - t0, 1),
        "device": image_gen.ImageGenerator.device,
    }


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
        "model": str(Engine.model_dir.name),
        "parameters_billion": round(Engine.model.num_parameters() / 1e9, 2),
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not Engine.load():
        return {"error": "模型未就绪，请先下载模型"}
    messages = [m.model_dump() for m in req.messages]
    prompt_ids = build_prompt_ids(Engine.tokenizer, messages, req.system_prompt)
    cfg = GenConfig(max_new_tokens=req.max_tokens, temperature=req.temperature,
                    top_p=req.top_p, top_k=req.top_k,
                    repetition_penalty=req.repetition_penalty)
    text, out_ids = generate(Engine.model, Engine.tokenizer, prompt_ids, cfg, Engine.device)
    return {"role": "assistant", "content": text, "tokens": len(out_ids)}


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    if not Engine.load():
        async def err_gen():
            yield f"data: {json.dumps({'error': '模型未就绪，请先下载模型'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")
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
