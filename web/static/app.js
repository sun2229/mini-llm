// 采样参数（可自行调整）
const PARAMS = {
  temperature: 0.7,
  top_p: 0.9,
  top_k: 50,
  max_tokens: 512,
  repetition_penalty: 1.1,
};

const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");
const statusEl = document.getElementById("status");
const welcomeEl = document.getElementById("welcome");

let messages = [];        // 发给后端的多轮历史
let generating = false;
let abortController = null;
let pollTimer = null;

// ---------------- 模型状态 ----------------
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.status === "ok") {
      statusEl.textContent = `${data.model} · ${data.device} · ${data.parameters_billion}B 参数`;
      document.getElementById("modelPanel").hidden = true;
      welcomeEl.hidden = false;
    } else {
      showModelPanel();
    }
  } catch {
    statusEl.textContent = "后端未连接";
  }
}
checkHealth();

function showModelPanel() {
  statusEl.textContent = "模型未安装";
  welcomeEl.hidden = true;
  const panel = document.getElementById("modelPanel");
  panel.hidden = false;
  fetch("/api/model-status")
    .then(r => r.json())
    .then(st => {
      if (st.gpu) {
        document.getElementById("gpuHint").hidden = false;
        document.getElementById("gpuHint").textContent =
          "检测到 NVIDIA 显卡，可稍后下载 GPU 加速组件提升速度";
      }
    })
    .catch(() => {});
}

// ---------------- 模型下载 ----------------
document.getElementById("downloadBtn").addEventListener("click", async () => {
  const btn = document.getElementById("downloadBtn");
  const wrap = document.getElementById("progressWrap");
  btn.disabled = true;
  btn.textContent = "下载中…";
  wrap.hidden = false;
  document.getElementById("modelPanelText").textContent =
    "正在从国内镜像下载开源大模型，约 0.6GB，请保持网络连接…";

  try {
    await fetch("/api/download-model", { method: "POST" });
  } catch { /* 轮询接口仍会返回进度 */ }

  pollTimer = setInterval(async () => {
    try {
      const r = await fetch("/api/download-progress");
      const st = await r.json();
      const pct = Math.round((st.progress || 0) * 100);
      document.getElementById("progressFill").style.width = pct + "%";
      document.getElementById("progressText").textContent =
        `${st.message || "下载中…"} ${pct}%`;

      if (st.done) {
        clearInterval(pollTimer);
        btn.disabled = false;
        btn.textContent = "正在加载模型…";
        document.getElementById("progressText").textContent = "下载完成，正在加载模型…";
        await loadModelAfterDownload();
      } else if (st.error) {
        clearInterval(pollTimer);
        btn.disabled = false;
        btn.textContent = "重试下载";
        document.getElementById("progressText").textContent = st.error;
        document.getElementById("modelPanelText").textContent = "下载失败，请检查网络后重试。";
      }
    } catch { /* 网络抖动忽略 */ }
  }, 1200);
});

async function loadModelAfterDownload() {
  for (let i = 0; i < 120; i++) {
    const r = await fetch("/api/health");
    const h = await r.json();
    if (h.status === "ok") {
      clearInterval(pollTimer);
      document.getElementById("modelPanel").hidden = true;
      welcomeEl.hidden = false;
      statusEl.textContent = `${h.model} · ${h.device} · ${h.parameters_billion}B 参数`;
      return;
    }
    await new Promise(res => setTimeout(res, 1000));
  }
  document.getElementById("progressText").textContent = "模型加载超时，请重试";
}

// ---------------- DOM 工具 ----------------
function addMessage(role, content = "") {
  if (!welcomeEl.hidden) welcomeEl.hidden = true;
  const panel = document.getElementById("modelPanel");
  if (!panel.hidden) panel.hidden = true;

  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "我" : "AI";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  // 复制按钮：聊天内容一键复制
  const copyBtn = document.createElement("button");
  copyBtn.className = "copy-btn";
  copyBtn.title = "复制这段内容";
  copyBtn.textContent = "复制";
  copyBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(bubble.textContent);
      copyBtn.textContent = "已复制";
      copyBtn.classList.add("copied");
      setTimeout(() => {
        copyBtn.textContent = "复制";
        copyBtn.classList.remove("copied");
      }, 1500);
    } catch {
      // 剪贴板不可用时退回选中文本
      const range = document.createRange();
      range.selectNodeContents(bubble);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      copyBtn.textContent = "已选中";
    }
  });
  msg.append(avatar, bubble, copyBtn);
  chatEl.appendChild(msg);
  chatEl.scrollTop = chatEl.scrollHeight;
  return bubble;
}

// ---------------- 发送与流式接收 ----------------
async function send() {
  const text = inputEl.value.trim();
  if (!text || generating) return;
  if (!document.getElementById("modelPanel").hidden) {
    showModelPanel();
    return;
  }

  messages.push({ role: "user", content: text });
  addMessage("user", text);
  inputEl.value = "";
  inputEl.style.height = "auto";

  const aiBubble = addMessage("ai");
  aiBubble.classList.add("cursor");

  generating = true;
  sendBtn.hidden = true;
  stopBtn.hidden = false;
  abortController = new AbortController();

  let full = "";
  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, ...PARAMS }),
      signal: abortController.signal,
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE 以两个换行分隔事件
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const evt of events) {
        const line = evt.replace(/^data: /, "").trim();
        if (!line) continue;
        const data = JSON.parse(line);
        if (data.done) {
          generating = false;
          break;
        }
        if (data.delta) {
          full += data.delta;
          aiBubble.textContent = full;
          chatEl.scrollTop = chatEl.scrollHeight;
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      full += "\n[已停止生成]";
    } else {
      full += `\n[请求失败：${err.message}]`;
    }
  } finally {
    aiBubble.classList.remove("cursor");
    aiBubble.textContent = full || "（无输出）";
    messages.push({ role: "assistant", content: full });
    generating = false;
    sendBtn.hidden = false;
    stopBtn.hidden = true;
    abortController = null;
  }
}

// ---------------- 事件绑定 ----------------
sendBtn.addEventListener("click", send);
stopBtn.addEventListener("click", () => abortController && abortController.abort());
clearBtn.addEventListener("click", () => {
  if (generating) return;
  messages = [];
  // 移除所有消息节点，保留 #welcome 与 #modelPanel
  [...chatEl.children].forEach(node => {
    if (node.id !== "welcome" && node.id !== "modelPanel") node.remove();
  });
  welcomeEl.hidden = false;
  welcomeEl.querySelector("h2").textContent = "对话已清空";
  welcomeEl.querySelector("p").textContent = "开始新的一轮对话吧。";
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

// 输入框自适应高度
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
});

// ================= 本地绘画模式 =================
const paintPanel = document.getElementById("paintPanel");
const tabChat = document.getElementById("tabChat");
const tabPaint = document.getElementById("tabPaint");
const paintBtn = document.getElementById("paintBtn");
const paintInput = document.getElementById("paintInput");
const paintHint = document.getElementById("paintHint");
const paintResult = document.getElementById("paintResult");
const paintImg = document.getElementById("paintImg");
const paintMeta = document.getElementById("paintMeta");
let paintPollTimer = null;
let lastPaintUrl = "";

function switchMode(mode) {
  tabChat.classList.toggle("active", mode === "chat");
  tabPaint.classList.toggle("active", mode === "paint");
  // 聊天区：隐藏 welcome/消息但保留绘画面板不冲突
  [...chatEl.children].forEach(node => {
    if (node.id === "paintPanel") return;
    node.hidden = mode === "paint";
  });
  paintPanel.hidden = mode !== "paint";
  clearBtn.hidden = mode !== "chat";
  if (mode === "paint") {
    checkPaintStatus();
  }
}

tabChat.addEventListener("click", () => switchMode("chat"));
tabPaint.addEventListener("click", () => switchMode("paint"));

async function checkPaintStatus() {
  try {
    const r = await fetch("/api/image-status");
    const st = await r.json();
    if (st.available) {
      paintHint.hidden = true;
      if (st.torch_cuda) {
        paintHint.hidden = false;
        paintHint.textContent = "已就绪 · GPU 加速";
      }
    } else {
      paintHint.hidden = false;
      paintHint.textContent = "绘画模型未下载（约 2GB，本地存储），首次生成前会自动下载。";
    }
  } catch { /* 忽略 */ }
}

// 绘画模型下载（复用统一进度接口）
async function downloadPaintModel() {
  paintHint.hidden = true;
  const wrap = document.getElementById("paintProgressWrap");
  wrap.hidden = false;
  document.getElementById("paintProgressText").textContent = "开始下载绘画模型（约 2GB）…";
  paintBtn.disabled = true;
  try {
    await fetch("/api/download-sd", { method: "POST" });
  } catch { /* 轮询兜底 */ }
  paintPollTimer = setInterval(async () => {
    try {
      const r = await fetch("/api/download-progress");
      const st = await r.json();
      const pct = Math.round((st.progress || 0) * 100);
      document.getElementById("paintProgressFill").style.width = pct + "%";
      document.getElementById("paintProgressText").textContent =
        `${st.message || "下载中…"} ${pct}%`;
      if (st.done) {
        clearInterval(paintPollTimer);
        paintBtn.disabled = false;
        document.getElementById("paintProgressText").textContent = "下载完成，开始生成…";
        await doPaint();
      } else if (st.error) {
        clearInterval(paintPollTimer);
        paintBtn.disabled = false;
        paintBtn.textContent = "重试下载";
        document.getElementById("paintProgressText").textContent = st.error;
      }
    } catch { /* 忽略 */ }
  }, 1200);
}

async function doPaint() {
  const prompt = paintInput.value.trim();
  if (!prompt) {
    paintHint.hidden = false;
    paintHint.textContent = "请先描述你想画的画面。";
    return;
  }
  paintBtn.disabled = true;
  paintBtn.textContent = "生成中…";
  paintHint.hidden = true;
  paintResult.hidden = true;
  try {
    const [w, h] = document.getElementById("paintSize").value.split("x").map(Number);
    const steps = parseInt(document.getElementById("paintSteps").value, 10);
    const r = await fetch("/api/gen-image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, width: w, height: h, steps }),
    });
    const data = await r.json();
    if (data.error) {
      if (data.download_required) {
        paintBtn.textContent = "生成图片";
        downloadPaintModel();
        return;
      }
      paintHint.hidden = false;
      paintHint.textContent = data.error;
      paintBtn.textContent = "生成图片";
      return;
    }
    lastPaintUrl = data.url;
    paintImg.src = data.url;
    paintMeta.textContent = `${prompt} · ${data.elapsed_s}s · 设备 ${data.device}`;
    paintResult.hidden = false;
  } catch (err) {
    paintHint.hidden = false;
    paintHint.textContent = `生成失败：${err.message}`;
  } finally {
    paintBtn.disabled = false;
    paintBtn.textContent = "生成图片";
  }
}

paintBtn.addEventListener("click", doPaint);
document.getElementById("paintAgainBtn").addEventListener("click", () => {
  paintImg.src = "";
  paintResult.hidden = true;
  doPaint();
});

// 下载图片到本地（另存为）
document.getElementById("paintSaveBtn").addEventListener("click", async () => {
  if (!lastPaintUrl) return;
  const fname = "star_" + Date.now() + ".png";
  try {
    const res = await fetch(lastPaintUrl);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fname;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch {
    const a = document.createElement("a");
    a.href = lastPaintUrl;
    a.download = fname;
    a.click();
  }
});
