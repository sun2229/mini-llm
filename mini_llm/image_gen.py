"""
本地图像生成（Stable Diffusion Turbo）：供 Star 桌面版使用。

设计：
  - 完全本地推理：模型下载到 models/sd/ 目录，不经过任何云端 API
  - 懒加载：首次调用才从本地加载 diffusers pipeline，避免拖慢启动
  - GPU 优先：检测到 CUDA 用显卡（fp16），否则回退 CPU（fp32，很慢）
  - 模型目录与文本模型隔离：models/sd/sd-turbo/，不会被文本引擎扫描误加载
"""

from __future__ import annotations

import random
import threading
from pathlib import Path

import torch

# 图像模型仓库与本地目录名
SD_REPO_ID = "AI-ModelScope/sd-turbo"
SD_DIRNAME = "sd-turbo"
SD_SUBDIR = "sd"  # 统一放在 models/sd/ 下


def sd_model_dir(models_dir: str | Path) -> Path:
    return Path(models_dir) / SD_SUBDIR / SD_DIRNAME


def sd_available(models_dir: str | Path) -> bool:
    d = sd_model_dir(models_dir)
    return d.is_dir() and (d / "model_index.json").exists()


class ImageGenerator:
    """单例式图像生成器。"""

    pipe = None
    device: str | None = None
    lock = threading.Lock()

    @classmethod
    def load(cls, models_dir: str | Path) -> bool:
        """加载本地 SD pipeline（幂等）。"""
        if cls.pipe is not None:
            return True
        if not sd_available(models_dir):
            return False
        with cls.lock:
            if cls.pipe is not None:
                return True
            from diffusers import AutoPipelineForText2Image  # 懒加载，打包时可排除

            cls.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if cls.device == "cuda" else torch.float32
            d = sd_model_dir(models_dir)
            try:
                cls.pipe = AutoPipelineForText2Image.from_pretrained(
                    str(d), torch_dtype=dtype, safety_checker=None,
                    variant="fp16")
                cls.pipe.to(cls.device)
                if cls.device == "cuda":
                    try:
                        cls.pipe.enable_attention_slicing()
                    except Exception:  # noqa: BLE001
                        pass
            except Exception as e:  # noqa: BLE001
                print(f"[错误] 图像模型加载失败: {e}")
                cls.pipe = None
                return False
        return True

    @classmethod
    def generate(cls, prompt: str, negative_prompt: str = "",
                 steps: int = 4, width: int = 512, height: int = 512,
                 seed: int | None = None):
        """生成一张图，返回 PIL.Image。"""
        if cls.pipe is None:
            raise RuntimeError("图像模型未加载")
        generator = torch.Generator(device=cls.device or "cpu")
        if seed is None:
            seed = random.randrange(0, 2**31)
        generator.manual_seed(seed)
        result = cls.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
            width=max(256, min(width, 768)),
            height=max(256, min(height, 768)),
            generator=generator,
            guidance_scale=0.0,  # Turbo 模型免引导
        )
        return result.images[0], seed
