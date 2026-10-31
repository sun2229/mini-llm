"""
模型管理：供 Star 桌面版使用。

职责：
  - 检测本地模型是否存在
  - 后台从 ModelScope 下载模型（可查询进度），纯 HTTP 实现，不依赖 modelscope 库
  - 检测 NVIDIA GPU（用于提示下载 GPU 加速组件）
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import urllib.request
from pathlib import Path

# 默认模型（下载包不含模型时，首次启动自动拉取）
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MODEL_DIRNAME = "Qwen2.5-0.5B-Instruct"

ALLOW_EXT = (".json", ".safetensors", ".txt", ".model")
DENY_KEYWORDS = ("pytorch_model", ".gguf", ".bin", ".msgpack", ".ot", ".h5")

_MODELSCOPE_API = "https://modelscope.cn/api/v1/models"


def _filter(fname: str) -> bool:
    return fname.endswith(ALLOW_EXT) and not any(k in fname for k in DENY_KEYWORDS)


def _api_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Star-LLM/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


class ModelManager:
    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._download_state: dict = {
            "running": False,
            "progress": 0.0,          # 0~1
            "current_file": "",
            "message": "",
            "done": False,
            "error": None,
        }
        self._lock = threading.Lock()

    # ------------------------------------------------------------- #
    # 状态查询
    # ------------------------------------------------------------- #
    def list_models(self) -> list[str]:
        return [d.name for d in self.models_dir.iterdir()
                if d.is_dir() and (d / "config.json").exists()]

    def has_model(self, name: str | None = None) -> bool:
        models = self.list_models()
        if not models:
            return False
        if name is None:
            return True
        return name in models

    def default_model_dir(self) -> Path:
        return self.models_dir / DEFAULT_MODEL_DIRNAME

    # ------------------------------------------------------------- #
    # GPU 检测：nvidia-smi 存在且能找到 GPU 即视为可用
    # ------------------------------------------------------------- #
    @staticmethod
    def has_nvidia_gpu() -> bool:
        try:
            r = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader")
            out = r.read().strip()
            r.close()
            return bool(out)
        except Exception:
            return False

    # ------------------------------------------------------------- #
    # 后台下载（纯 HTTP）
    # ------------------------------------------------------------- #
    def download_default(self, repo_id: str | None = None) -> dict:
        repo_id = repo_id or DEFAULT_MODEL_ID
        with self._lock:
            if self._download_state["running"]:
                return self._download_state.copy()
            self._download_state.update(
                running=True, progress=0.0, current_file="",
                message="开始下载", done=False, error=None)

        def _worker():
            target_dir = self.models_dir / repo_id.split("/")[-1]
            try:
                files = self._planned_files(repo_id)
                if not files:
                    raise RuntimeError("无法获取模型文件清单，请检查网络")
                total = sum(files.values())
                done_bytes = 0

                for f in files:
                    size = files[f]
                    rel = f.lstrip("/")
                    local = target_dir / rel
                    local.parent.mkdir(parents=True, exist_ok=True)

                    def _on_progress(inc: int, name: str = rel):
                        nonlocal done_bytes
                        done_bytes += inc
                        with self._lock:
                            self._download_state.update(
                                current_file=name,
                                progress=min(1.0, done_bytes / max(total, 1)),
                                message=f"下载中 {name}")

                    self._download_one(repo_id, f, local, size, _on_progress)

                # 完整性兜底：config.json 缺失则报错
                if not (target_dir / "config.json").exists():
                    raise RuntimeError("模型文件不完整（缺少 config.json）")

                with self._lock:
                    self._download_state.update(
                        running=False, progress=1.0, done=True,
                        message="下载完成")
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._download_state.update(
                        running=False, done=False, error=str(e),
                        message=f"下载失败: {e}")

        threading.Thread(target=_worker, daemon=True).start()
        return self._download_state.copy()

    def _planned_files(self, repo_id: str) -> dict[str, int]:
        """通过 ModelScope 公开 API 获取文件清单 {路径: 字节数}。"""
        url = f"{_MODELSCOPE_API}/{repo_id}/repo/files?Revision=master&Recursive=true"
        data = _api_json(url)
        files_data = (data.get("Data") or {}).get("Files") or []
        picked: dict[str, int] = {}
        for f in files_data:
            name = f.get("Path", "")
            size = int(f.get("Size") or 0)
            if _filter(name):
                picked[name] = size
        return picked

    def _download_one(self, repo_id: str, file_path: str, local: Path,
                      total_size: int, on_progress) -> None:
        """带断点续传的下载。"""
        import requests

        if total_size and local.exists() and local.stat().st_size >= total_size:
            return  # 已完整存在
        url = (f"{_MODELSCOPE_API}/{repo_id}/repo?FilePath={file_path}"
               "&Revision=master")
        tmp = local.with_suffix(local.suffix + ".part")
        # 断点续传：.part 已完整时直接改名
        if tmp.exists() and total_size and tmp.stat().st_size >= total_size:
            for _ in range(10):
                try:
                    tmp.replace(local)
                    return
                except OSError:
                    time.sleep(2)
            raise OSError(f"文件写入被占用: {local}")
        headers = {}
        if tmp.exists():
            headers["Range"] = f"bytes={tmp.stat().st_size}-"
        try:
            with requests.get(url, stream=True, timeout=(30, 120),
                              headers=headers, allow_redirects=True) as r:
                r.raise_for_status()
                # 仅当服务器真正返回 206（Partial Content）才续传追加，
                # 否则从头写入，避免 CDN 忽略 Range 导致文件损坏
                resume = bool(headers) and tmp.exists() and r.status_code == 206
                mode = "ab" if resume else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                            on_progress(len(chunk))
        except requests.exceptions.RequestException:
            # 416 说明 .part 已完整，删除后全量重下
            if tmp.exists() and total_size and tmp.stat().st_size >= total_size:
                for _ in range(10):
                    try:
                        tmp.replace(local)
                        return
                    except OSError:
                        time.sleep(2)
                raise OSError(f"文件写入被占用: {local}")
            raise
        # Windows 下杀毒/索引可能短暂占用文件，重命名加重试
        for _ in range(10):
            try:
                tmp.replace(local)
                return
            except OSError:
                time.sleep(2)
        raise OSError(f"文件写入被占用: {local}")

    def download_progress(self) -> dict:
        with self._lock:
            return self._download_state.copy()

    def clear_models(self) -> int:
        """删除全部本地模型，返回删除的目录数。"""
        count = 0
        for d in self.models_dir.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                count += 1
        return count
