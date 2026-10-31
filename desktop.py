"""
Star 桌面版启动器：双击 Star.exe 后弹出独立桌面窗口（内嵌聊天界面），
同时本地服务在浏览器里也可访问。

流程：
  1. 启动内置 FastAPI 服务（线程内运行 uvicorn）
  2. 等待服务就绪
  3. 打开 pywebview 桌面窗口（EdgeChromium/WebView2 内核）
  4. 窗口关闭时退出整个进程
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

APP_TITLE = "Star · 本地大模型"


def _pick_port(start: int = 8123, tries: int = 20) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 8123


def _free_port(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def start_server(port: int):
    try:
        import uvicorn
        # 直接传 app 对象，避免打包后字符串导入（"web.app:app"）找不到模块
        import web.app
        config = uvicorn.Config(web.app.app, host="127.0.0.1", port=port,
                                log_level="warning", access_log=False)
        server = uvicorn.Server(config)
        server.run()
    except Exception:
        import traceback
        try:
            log_path = os.path.join(os.environ.get("TEMP", "."), "star_server_error.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        raise


def _wait_ready(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _free_port(port):
            try:
                import urllib.request
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health",
                                            timeout=2) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
        time.sleep(0.4)
    return False


def main():
    # 若已有一个实例在跑，直接复用那个服务（避免多开）
    port = _pick_port()
    server_thread = threading.Thread(target=start_server, args=(port,),
                                     daemon=True)
    server_thread.start()
    if not _wait_ready(port):
        print("[错误] 内置服务启动超时")
        return 1

    url = f"http://127.0.0.1:{port}/"
    print(f"[Star] 服务已启动: {url}")

    use_window = os.environ.get("STAR_WINDOW", "1") != "0"
    if not use_window:
        webbrowser.open(url)
        print("[Star] 已在浏览器打开，关闭浏览器后请按 Ctrl+C 退出")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return 0

    try:
        import webview  # noqa: WPS433

        window = webview.create_window(APP_TITLE, url,
                                       width=980, height=700,
                                       min_size=(720, 520))
        webview.start(debug=False)
    except Exception as e:  # noqa: BLE001
        # 桌面窗口不可用时回退到浏览器
        print(f"[提示] 桌面窗口不可用（{e}），改用浏览器打开")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return 0
    return 0


if __name__ == "__main__":
    # 打包后 cwd 可能被改，固定到 exe 所在目录，保证 models/、web/ 相对路径稳定
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).parent)
        # windowed 模式无控制台，把输出/异常重定向到日志文件便于诊断
        try:
            log_path = os.path.join(os.environ.get("TEMP", "."), "star_app.log")
            sys.stdout = sys.stderr = open(log_path, "w", encoding="utf-8")
        except Exception:
            pass
    sys.exit(main())
