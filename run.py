"""
项目启动入口：自动加载模型并启动 Web 服务。

用法：
  python run.py
  python run.py --model models/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000
"""

import argparse
import os
from pathlib import Path

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="启动 Mini-LLM Web 服务")
    parser.add_argument("--model", default=None,
                        help="模型目录，默认 models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.model:
        os.environ["MINI_LLM_MODEL"] = str(Path(args.model).resolve())

    # uvicorn 以导入路径方式加载 app
    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
