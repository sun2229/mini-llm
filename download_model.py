"""
模型下载脚本：优先使用 ModelScope（国内网络快、无需特殊网络配置），
回退到 HuggingFace。只下载推理必需文件，避免拉取无关权重。

用法：
  python download_model.py                      # 默认下载 Qwen2.5-1.5B-Instruct
  python download_model.py --model 0.5B        # 下载更小的 0.5B 版本
  python download_model.py --source huggingface
"""

import argparse
import sys

# 模型名映射
MODELS = {
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
}

# 推理必需文件（跳过 .gguf、.pt、pytorch 二进制等冗余文件）
ALLOW_EXT = (".json", ".safetensors", ".txt", ".model")
DENY_KEYWORDS = ("pytorch_model", ".gguf", ".bin", ".msgpack", ".ot", ".h5")


def _filter_files(files: list[str]) -> list[str]:
    picked = []
    for f in files:
        if any(k in f for k in DENY_KEYWORDS):
            continue
        if f.endswith(ALLOW_EXT):
            picked.append(f)
    return picked


def from_modelscope(repo_id: str, out_dir: str):
    from modelscope import snapshot_download
    path = snapshot_download(repo_id, local_dir=out_dir,
                             allow_file_pattern=lambda f: f.endswith(ALLOW_EXT)
                             and not any(k in f for k in DENY_KEYWORDS))
    print(f"[ModelScope] 已下载到 {path}")


def from_huggingface(repo_id: str, out_dir: str):
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id, local_dir=out_dir,
                             allow_patterns=[f"*{e}" for e in ALLOW_EXT],
                             ignore_patterns=[f"*{k}*" for k in DENY_KEYWORDS])
    print(f"[HuggingFace] 已下载到 {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), default="1.5B")
    parser.add_argument("--source", choices=["modelscope", "huggingface"],
                        default="modelscope")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    repo_id = MODELS[args.model]
    out_dir = args.out or f"models/{repo_id.split('/')[-1]}"

    print(f"准备下载 {repo_id} -> {out_dir}（来源：{args.source}）")
    try:
        if args.source == "modelscope":
            from_modelscope(repo_id, out_dir)
        else:
            from_huggingface(repo_id, out_dir)
    except ImportError:
        print(f"未安装 {args.source} 依赖，请先执行: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"下载失败: {e}")
        print("可尝试切换来源：python download_model.py --source huggingface")
        sys.exit(1)


if __name__ == "__main__":
    main()
