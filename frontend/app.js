// ---- 工具 ----
const $ = (sel) => document.querySelector(sel);

function genId() {
  return "c-" + Math.random().toString(36).slice(2, 8) + Date.now().toString(36);
}

function escapeHtml(s) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return String(s).replace(/[&<>"']/g, (c) => map[c]);
}

// ---- 状态 ----
let conversationId = genId();

const messagesEl = $("#messages");
const formEl = $("#chat-form");
const inputEl = $("#message-input");
const docListEl = $("#doc-list");
const dropZoneEl = $("#drop-zone");
const fileInputEl = $("#file-input");
const statusEl = $("#upload-status");

// ---- 文档列表 ----
async function refreshDocuments() {
  const res = await fetch("/api/documents");
  const docs = await res.json();
  docListEl.innerHTML = "";
  if (!docs.length) {
    docListEl.innerHTML = '<li class="empty">还没有文档,先上传一个</li>';
    return;
  }
  for (const d of docs) {
    const li = document.createElement("li");
    li.className = "doc-item";
    li.innerHTML =
      `<span class="doc-name" title="${escapeHtml(d.name)}">${escapeHtml(d.name)}</span>` +
      `<span class="doc-count">${d.chunks} 片段</span>`;
    docListEl.appendChild(li);
  }
}

// ---- 上传 ----
function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = "status " + (kind || "");
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList);
  const allowed = files.filter((f) => /\.(md|markdown|txt|pdf)$/i.test(f.name));
  if (!allowed.length) {
    setStatus("只支持 .md / .txt / .pdf 文件", "error");
    return;
  }
  setStatus(`上传中 ${allowed.length} 个文件...`, "info");
  const fd = new FormData();
  allowed.forEach((f) => fd.append("files", f));
  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    const parts = [];
    if (data.uploaded.length) parts.push(`已索引 ${data.uploaded.length} 个文件`);
    if (data.skipped.length) parts.push(`跳过 ${data.skipped.length} 个`);
    parts.push(`共 ${data.total_chunks} 个片段`);
    setStatus(parts.join(" · "), "ok");
    refreshDocuments();
  } catch (e) {
    setStatus("上传失败:" + e.message, "error");
  }
}

dropZoneEl.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZoneEl.classList.add("dragging");
});
dropZoneEl.addEventListener("dragleave", () => dropZoneEl.classList.remove("dragging"));
dropZoneEl.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZoneEl.classList.remove("dragging");
  uploadFiles(e.dataTransfer.files);
});

$("#pick-btn").addEventListener("click", () => fileInputEl.click());
fileInputEl.addEventListener("change", () => {
  uploadFiles(fileInputEl.files);
  fileInputEl.value = "";
});

// ---- 聊天 ----
function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addMessage(role, text, sources) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);

  if (sources && sources.length) {
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = `来源 (${sources.length})`;
    details.appendChild(summary);
    for (const s of sources) {
      const p = document.createElement("div");
      p.className = "source";
      p.textContent = `[${s.score}] ${s.section || s.source} — ${s.text}`;
      details.appendChild(p);
    }
    wrap.appendChild(details);
  }

  messagesEl.appendChild(wrap);
  scrollToBottom();
}

function addThinking() {
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble thinking";
  bubble.textContent = "思考中...";
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  addMessage("user", text);

  const thinking = addThinking();
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, conversation_id: conversationId }),
    });
    const data = await res.json();
    if (data.conversation_id) conversationId = data.conversation_id;
    thinking.remove();
    addMessage("assistant", data.answer, data.sources);
  } catch (err) {
    thinking.remove();
    addMessage("assistant", "出错了:" + err.message);
  }
});

// ---- 新对话 ----
$("#new-chat-btn").addEventListener("click", () => {
  conversationId = genId();
  messagesEl.innerHTML = "";
  const welcome = document.createElement("div");
  welcome.className = "welcome";
  welcome.innerHTML = "<p>👋 已开始新对话</p>";
  messagesEl.appendChild(welcome);
});

// ---- 启动 ----
refreshDocuments();
