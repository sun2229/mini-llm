# Mini-LLM：从零手搓本地大模型

一个**不依赖 transformers 模型类**、从零手写的 Decoder-Only Transformer 推理与训练引擎，
架构对齐 **Qwen2.5**，可以直接加载官方开源权重，在你自己的电脑上跑起来。
附带 FastAPI 后端和浏览器聊天界面，支持流式输出、多轮对话和从零训练。

> 运行环境检测：RTX 4060 Laptop（8GB 显存）+ i7-14650HX + 16GB 内存
> 推荐模型：**Qwen2.5-1.5B-Instruct**（fp16 约 3GB，8GB 显存轻松运行）

---

## 一、目录结构

```
mini-llm/
├── mini_llm/
│   ├── model.py        # 核心：RMSNorm / RoPE / GQA / SwiGLU / KV Cache 全部手写
│   ├── weights.py      # 手动读取 safetensors 官方权重并映射
│   ├── tokenizer.py    # 底层 tokenizers 加载 tokenizer.json
│   ├── chat.py         # Qwen ChatML 对话模板
│   ├── generation.py   # 采样器（温度/top-k/top-p/重复惩罚）+ 流式生成
│   └── train.py        # 从零训练循环（AdamW + cosine LR + 梯度裁剪）
├── web/
│   ├── app.py          # FastAPI：REST + SSE 流式接口
│   └── static/         # 浏览器聊天页面（HTML/CSS/JS）
├── scripts/
│   ├── chat_cli.py     # 命令行多轮对话
│   ├── train_tiny.py   # 从零训练迷你模型示例
│   └── smoke_test.py   # 冒烟测试（含 KV Cache 一致性验证）
├── data/sample.txt     # 示例语料
├── download_model.py   # 模型下载（ModelScope 优先，国内速度快）
├── run.py              # 一键启动 Web 服务
└── requirements.txt
```

---

## 二、安装

建议使用 Python 3.10 ~ 3.13。

### 1. 安装 PyTorch

**有 NVIDIA 显卡（推荐，CUDA 加速）：**

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**无独显 / 纯 CPU：**

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. 安装其余依赖

```powershell
pip install -r requirements.txt
```

---

## 三、下载模型

国内网络优先使用 ModelScope，无需特殊网络配置：

```powershell
# 默认下载 Qwen2.5-1.5B-Instruct（推荐）
python download_model.py

# 想更快、更省空间可下载 0.5B 版本
python download_model.py --model 0.5B

# 显存宽裕可下载 3B 版本（fp16 约 6GB）
python download_model.py --model 3B
```

模型会保存到 `models/` 目录，脚本只下载推理必需文件。

---

## 四、使用方式

### 1. Web 聊天界面（推荐）

```powershell
python run.py
```

浏览器打开 <http://127.0.0.1:8000>，即可像本地版 ChatGPT 一样对话，
回答逐字流式输出，支持停止生成与清空对话。

指定其他模型：

```powershell
python run.py --model models/Qwen2.5-0.5B-Instruct --port 8000
```

### 2. 命令行对话

```powershell
python scripts/chat_cli.py --model models/Qwen2.5-1.5B-Instruct
```

输入 `clear` 清空历史，`exit` 退出。

### 3. 从零训练自己的模型

```powershell
# 先准备一个文本文件（默认 data/sample.txt），然后：
python scripts/train_tiny.py --data data/sample.txt `
        --tokenizer models/Qwen2.5-1.5B-Instruct --out checkpoints/tiny
```

训练器会初始化一个约千万参数的迷你模型，在你的语料上训练并保存 safetensors 检查点。

---

## 五、核心原理速览

| 组件 | 作用 | 本项目实现 |
| --- | --- | --- |
| RMSNorm | 稳定训练的层归一化 | `model.py: RMSNorm` |
| RoPE | 把位置信息旋转编码进 Q/K | `model.py: precompute_rope / apply_rope` |
| GQA | 多组 Q 共享 K/V，节省缓存 | `model.py: Attention / repeat_kv` |
| SwiGLU | SiLU 门控前馈网络 | `model.py: FeedForward` |
| KV Cache | 缓存历史 K/V，增量解码 | `model.py` + `generation.py` |
| 采样 | 温度 / top-k / top-p / 重复惩罚 | `generation.py` |
| 权重加载 | 官方权重映射到自研模块 | `weights.py` |

**一次完整生成的流程：**

1. **Prefill**：整段 prompt 一次性喂入模型，并行建立每一层的 KV Cache；
2. 对最后一个位置的 logits 施加采样策略，得到下一个 token；
3. **Decode**：之后每次只喂入一个新 token，复用缓存避免重复计算，循环直到输出
   `<|im_end|>` 或达到 `max_tokens`。

---

## 六、测试

```powershell
python scripts/smoke_test.py
```

冒烟测试无需下载大模型，会验证：前向输出形状、**KV Cache 增量解码与一次性 prefill
结果一致**、采样器合法性、生成循环正常结束。

---

## 七、常见问题

- **显存不足（OOM）**：换用 `0.5B` 模型，或降低 `max_tokens`；纯 CPU 也能运行只是更慢。
- **ModelScope 下载失败**：加 `--source huggingface` 切换来源。
- **首次推理较慢**：CUDA 内核需要预热，后续生成会明显变快。
- **回答重复**：调高 `repetition_penalty`（如 1.15）或降低 `temperature`。

---

## 八、推送到你自己的 GitHub

```powershell
git init
git add .
git commit -m "feat: hand-crafted local LLM engine"
gh repo create my-mini-llm --public --source . --push
```

> 注意：把 `models/`、`checkpoints/` 加入 `.gitignore`，不要把几十 GB 的权重提交到仓库。
