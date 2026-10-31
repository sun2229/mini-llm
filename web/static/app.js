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
  msg.append(avatar, bubble);
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
