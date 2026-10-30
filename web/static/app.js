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

let messages = [];        // 发给后端的多轮历史
let generating = false;
let abortController = null;

// ---------------- 模型状态 ----------------
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.status === "ok") {
      statusEl.textContent = `${data.model} · ${data.device} · ${data.parameters_billion}B 参数`;
    } else {
      statusEl.textContent = "未找到模型，请先运行 download_model.py 下载";
    }
  } catch {
    statusEl.textContent = "后端未连接";
  }
}
checkHealth();

// ---------------- DOM 工具 ----------------
function addMessage(role, content = "") {
  const welcome = chatEl.querySelector(".welcome");
  if (welcome) welcome.remove();

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
  chatEl.innerHTML = "";
  const welcome = document.createElement("div");
  welcome.className = "welcome";
  welcome.innerHTML = "<h2>对话已清空</h2><p>开始新的一轮对话吧。</p>";
  chatEl.appendChild(welcome);
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
