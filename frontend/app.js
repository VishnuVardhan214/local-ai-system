// Plain JavaScript, no external files, so the page works fully offline.
const $ = (id) => document.getElementById(id);
let sessionId = localStorage.getItem("lai_session") || null;
let busy = false;

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Very small and safe renderer: text is escaped first, then code and bold are added.
function render(text) {
  let out = esc(text);
  out = out.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => `<pre><code>${code}</code></pre>`);
  out = out.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  return out.replace(/\n/g, "<br>").replace(/<pre><code>([\s\S]*?)<\/code><\/pre>/g, (m) => m.replace(/<br>/g, "\n"));
}

function addMessage(role, label) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.innerHTML = '<div class="who"></div><div class="body"></div>';
  div.querySelector(".who").textContent = label;
  $("chat").appendChild(div);
  $("chat").scrollTop = $("chat").scrollHeight;
  return div;
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return r.json();
}

// ---- chat streaming -------------------------------------------------------
async function send(text) {
  if (busy || !text.trim()) return;
  busy = true; $("sendBtn").disabled = true;
  addMessage("user", "You").querySelector(".body").textContent = text;
  let bot = addMessage("assistant", "Assistant");
  let raw = "";
  const body = bot.querySelector(".body");
  try {
    const resp = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId, agent: $("agentSelect").value }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const line = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (line.startsWith("data: ")) {
          raw = handle(JSON.parse(line.slice(6)), bot, body, raw);
        }
      }
    }
  } catch (e) {
    const p = document.createElement("div"); p.className = "err"; p.textContent = "Error: " + e.message;
    body.appendChild(p);
  }
  busy = false; $("sendBtn").disabled = false; setBadge(null);
  loadSessions(); loadDocs();
}

function handle(ev, bot, body, raw) {
  switch (ev.type) {
    case "session": sessionId = ev.id; localStorage.setItem("lai_session", sessionId); break;
    case "route":
      bot.querySelector(".who").textContent = ev.title + " agent (" + ev.reason + ")";
      setBadge(ev.title, ev.agent); break;
    case "status": $("agentStep").textContent = ev.text; break;
    case "token": raw += ev.text; body.innerHTML = render(raw); $("chat").scrollTop = $("chat").scrollHeight; break;
    case "tool_call": logTool("call", ev.name + " " + JSON.stringify(ev.args).slice(0, 140)); break;
    case "tool_result": logTool(ev.ok ? "ok" : "fail", ev.name + ": " + ev.preview.slice(0, 160)); break;
    case "approval_request": showApproval(ev); break;
    case "approval_result": clearApproval(ev.id, ev.approved); break;
    case "sources": showSources(bot, ev.items); break;
    case "stats":
      if (ev.eval_count) {
        let m = bot.querySelector(".meta");
        if (!m) { m = document.createElement("div"); m.className = "meta"; bot.appendChild(m); }
        m.textContent = ev.eval_count + " tokens, " + ev.tokens_per_s + " tok/s" + (ev.ttft_s ? ", first token " + ev.ttft_s + "s" : "");
      } break;
    case "error": { const p = document.createElement("div"); p.className = "err"; p.textContent = ev.message; body.appendChild(p); break; }
    case "done": break;
  }
  return raw;
}

function showSources(bot, items) {
  const box = document.createElement("div"); box.className = "sources";
  box.innerHTML = "<strong>Sources</strong>";
  items.forEach((s) => {
    const d = document.createElement("div");
    d.textContent = "[" + s.n + "] " + s.doc + ", " + s.locator + ": " + s.snippet;
    box.appendChild(d);
  });
  bot.appendChild(box);
}

function setBadge(title, name) {
  const b = $("agentBadge");
  b.textContent = title || "idle"; b.className = "badge" + (title ? "" : " idle");
  if (!title) $("agentStep").textContent = "";
}

function logTool(kind, text) {
  const li = document.createElement("li");
  li.className = kind === "ok" ? "ok" : kind === "fail" ? "fail" : "";
  li.textContent = (kind === "call" ? "-> " : kind === "ok" ? "ok " : "fail ") + text;
  $("toolLog").prepend(li);
}

// ---- approvals ----------------------------------------------------------------
function showApproval(ev) {
  if ($("approvals").textContent === "Nothing waiting.") $("approvals").textContent = "";
  const box = document.createElement("div"); box.className = "approval"; box.id = "ap_" + ev.id;
  box.innerHTML = "<strong></strong><div class='d'></div><div class='row'><button class='primary'>Approve</button><button>Deny</button></div>";
  box.querySelector("strong").textContent = ev.reason;
  box.querySelector(".d").textContent = JSON.stringify(ev.args).slice(0, 400);
  const [yes, no] = box.querySelectorAll("button");
  yes.onclick = () => decide(ev.id, true); no.onclick = () => decide(ev.id, false);
  $("approvals").appendChild(box);
}
async function decide(id, approve) {
  try { await api("/api/approve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id, approve }) }); } catch (e) {}
}
function clearApproval(id, approved) {
  const box = $("ap_" + id); if (box) box.remove();
  logTool(approved ? "ok" : "fail", approved ? "action approved" : "action denied");
  if (!$("approvals").children.length) $("approvals").textContent = "Nothing waiting.";
}

// ---- sessions, documents, models --------------------------------------------------
async function loadSessions() {
  const list = await api("/api/sessions");
  const ul = $("sessionList"); ul.innerHTML = "";
  list.forEach((s) => {
    const li = document.createElement("li"); if (s.id === sessionId) li.className = "active";
    const t = document.createElement("span"); t.textContent = s.title || "(untitled)";
    const x = document.createElement("span"); x.className = "x"; x.textContent = "x";
    li.append(t, x);
    t.onclick = () => openSession(s.id);
    x.onclick = async (e) => { e.stopPropagation(); await api("/api/sessions/" + s.id, { method: "DELETE" });
      if (s.id === sessionId) newChat(); loadSessions(); };
    ul.appendChild(li);
  });
}
async function openSession(id) {
  sessionId = id; localStorage.setItem("lai_session", id);
  $("chat").innerHTML = "";
  const msgs = await api("/api/sessions/" + id);
  msgs.forEach((m) => {
    const label = m.role === "user" ? "You" : (m.agent ? m.agent + " agent" : "Assistant");
    const div = addMessage(m.role === "user" ? "user" : "assistant", label);
    div.querySelector(".body").innerHTML = m.role === "user" ? esc(m.content) : render(m.content);
    if (m.meta && m.meta.sources && m.meta.sources.length) showSources(div, m.meta.sources);
  });
  loadSessions();
}
function newChat() { sessionId = null; localStorage.removeItem("lai_session"); $("chat").innerHTML = ""; $("toolLog").innerHTML = ""; loadSessions(); }

async function loadDocs() {
  const d = await api("/api/documents");
  const ul = $("docList"); ul.innerHTML = "";
  d.documents.forEach((doc) => {
    const li = document.createElement("li");
    const t = document.createElement("span"); t.textContent = doc.name + " (" + doc.n_chunks + ")";
    const x = document.createElement("span"); x.className = "x"; x.textContent = "x";
    x.onclick = async () => { await api("/api/documents/" + doc.id, { method: "DELETE" }); loadDocs(); };
    li.append(t, x); ul.appendChild(li);
  });
}
$("fileInput").addEventListener("change", async (e) => {
  const files = e.target.files; if (!files.length) return;
  const fd = new FormData(); for (const f of files) fd.append("files", f);
  $("uploadMsg").textContent = "Indexing...";
  try {
    const res = await api("/api/upload", { method: "POST", body: fd });
    $("uploadMsg").textContent = res.map((r) => r.name + ": " + (r.status === "added" ? r.chunks + " passages" : r.status + (r.error ? " (" + r.error + ")" : ""))).join("; ");
  } catch (err) { $("uploadMsg").textContent = "Upload failed: " + err.message; }
  e.target.value = ""; loadDocs();
});

async function loadAgents() {
  const agents = await api("/api/agents");
  const sel = $("agentSelect"); sel.innerHTML = "";
  const auto = document.createElement("option"); auto.value = "auto"; auto.textContent = "Auto (router decides)"; sel.appendChild(auto);
  agents.forEach((a) => { const o = document.createElement("option"); o.value = a.name; o.textContent = a.title; o.title = a.description; sel.appendChild(o); });
}
async function loadModels() {
  const m = await api("/api/models");
  const sel = $("modelSelect"); sel.innerHTML = "";
  const names = m.available.length ? m.available : [m.current];
  if (!names.includes(m.current)) names.unshift(m.current);
  names.forEach((n) => { const o = document.createElement("option"); o.value = n; o.textContent = n; if (n === m.current) o.selected = true; sel.appendChild(o); });
}
$("modelSelect").addEventListener("change", async (e) => {
  try { await api("/api/model", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: e.target.value }) }); }
  catch (err) { alert(err.message); loadModels(); }
});
$("unloadBtn").onclick = () => api("/api/unload", { method: "POST" });

// ---- resource monitor ----------------------------------------------------------------
function meter(label, used, total, unit) {
  const pct = total ? Math.min(100, Math.round((used / total) * 100)) : 0;
  const cls = pct > 90 ? "bad" : pct > 75 ? "warn" : "";
  return `<div class="meter"><div>${label}: ${used}${unit} / ${total}${unit} (${pct}%)</div><div class="bar"><div class="fill ${cls}" style="width:${pct}%"></div></div></div>`;
}
async function pollResources() {
  try {
    const r = await api("/api/resources");
    let html = meter("CPU", Math.round(r.cpu_percent), 100, "%");
    html += meter("RAM", r.ram_used_mb, r.ram_total_mb, " MB");
    if (r.swap_total_mb) html += meter("Swap", r.swap_used_mb, r.swap_total_mb, " MB");
    html += r.gpu ? meter("VRAM", r.gpu.vram_used_mb, r.gpu.vram_total_mb, " MB") : '<div class="note">No NVIDIA GPU detected.</div>';
    $("meters").innerHTML = html;
    $("loadedModels").textContent = r.loaded_models.length
      ? "Loaded: " + r.loaded_models.map((m) => m.name + " (" + m.vram_mb + " MB on GPU)").join(", ") : "No model loaded.";
    if (!busy && r.state && r.state.status !== "idle") setBadge(r.state.agent || r.state.status);
  } catch (e) {}
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); const t = $("input").value; $("input").value = ""; send(t); });
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("form").requestSubmit(); } });
$("newChat").onclick = newChat;

loadAgents(); loadModels(); loadSessions(); loadDocs(); pollResources(); setInterval(pollResources, 2500);
if (sessionId) openSession(sessionId).catch(() => newChat());
