# -*- mode: python ; coding: utf-8 -*-
"""Star 桌面版 PyInstaller 打包配置（onedir 模式）。"""

import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
    copy_metadata,
)

# 项目根目录
ROOT = SPECPATH

# 图像生成依赖：diffusers / transformers 动态导入多，全量收集
_transformers_datas = collect_data_files("transformers")
_transformers_hidden = collect_submodules("transformers")
_diffusers_datas = collect_data_files("diffusers")
_diffusers_hidden = collect_submodules("diffusers")
_hub_datas = collect_data_files("huggingface_hub")
# 依赖版本检查需要包的 dist-info 元数据（importlib.metadata）
_meta = (
    copy_metadata("requests")
    + copy_metadata("transformers")
    + copy_metadata("diffusers")
    + copy_metadata("accelerate")
    + copy_metadata("huggingface_hub")
    + copy_metadata("safetensors")
    + copy_metadata("tokenizers")
    + copy_metadata("torch")
    + copy_metadata("numpy")
    + copy_metadata("Pillow")
)

a = Analysis(
    [os.path.join(ROOT, "desktop.py")],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (os.path.join(ROOT, "web", "static"), "web/static"),
        *_transformers_datas,
        *_diffusers_datas,
        *_hub_datas,
        *_meta,
    ],
    hiddenimports=[
        # uvicorn 动态导入的子模块
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.loops.asyncio",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.websockets_impl",
        "uvicorn.protocols.utils",
        "uvicorn.lifespan.on",
        "uvicorn.lifespan.off",
        "uvicorn._subprocess",
        # FastAPI / 模型后端
        "web.app",
        "mini_llm",
        "mini_llm.model",
        "mini_llm.weights",
        "mini_llm.tokenizer",
        "mini_llm.chat",
        "mini_llm.generation",
        "mini_llm.model_manager",
        "mini_llm.image_gen",
        # 图像生成（本地绘画）
        "diffusers",
        "diffusers.pipelines.stable_diffusion",
        "diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion",
        "transformers",
        "transformers.models.clip",
        "transformers.models.clip.tokenization_clip",
        "transformers.models.clip.modeling_clip",
        "huggingface_hub",
        *_transformers_hidden,
        *_diffusers_hidden,
        # 桌面窗口
        "webview",
        "webview.platforms",
        "webview.platforms.edgechromium",
        # tokenizers / safetensors 的 Rust 绑定
        "tokenizers",
        "safetensors",
        "safetensors.torch",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "modelscope",
        "torchvision",
        "matplotlib",
        "scipy",
        "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Star",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # 无控制台窗口，双击即弹软件窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Star",
)
