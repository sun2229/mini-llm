# -*- mode: python ; coding: utf-8 -*-
"""Star 桌面版 PyInstaller 打包配置（onedir 模式）。"""

import os

from PyInstaller.utils.hooks import collect_data_files

# 项目根目录
ROOT = SPECPATH

a = Analysis(
    [os.path.join(ROOT, "desktop.py")],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (os.path.join(ROOT, "web", "static"), "web/static"),
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
        "transformers",
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
