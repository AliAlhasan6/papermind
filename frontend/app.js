// PaperMind frontend — Day 6: live chat + sources + KG viewer
const API = "";  // same origin — served by FastAPI

const $ = (id) => document.getElementById(id);
let pendingFiles = [];

// ---------- Ingestion ----------
$("pdf-input").addEventListener("change", (e) => {
  pendingFiles = Array.from(e.target.files);
  $("ingest-btn").disabled = pendingFiles.length === 0;
  $("ingest-status").textContent = pendingFiles.length
    ? pendingFiles.length + " file(s) selected" : "";
});

$("ingest-btn").addEventListener("click", async () => {
  if (!pendingFiles.length) return;
  $("ingest-btn").disabled = true;
  $("ingest-status").textContent = "ingesting... (may take a minute)";
  const fd = new FormData();
  pendingFiles.forEach((f) => fd.append("files", f));
  try {
    const r = await fetch(API + "/ingest", { method: "POST", body: fd });
    const data = await r.json();
    $("ingest-status").textContent =
      "done: " + data.papers + " papers, " + data.chunks + " chunks";
    refreshLibrary();
    pendingFiles = [];
    $("pdf-input").value = "";
  } catch (err) {
    $("ingest-status").textContent = "error: " + err.message;
  }
});

async function refreshLibrary() {
  try {
    const r = await fetch(API + "/papers");
    const data = await r.json();
    $("paper-list").innerHTML = data.papers.length
      ? data.papers.map((p) =>
          "<li>" + p.name + "<span class='meta'>" + p.chunks + " ch</span></li>").join("")
      : "<li><em>empty</em></li>";
  } catch {
    $("paper-list").innerHTML = "<li><em>API offline</em></li>";
  }
}

// ---------- Chat ----------
$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("chat-input").value.trim();
  if (!q) return;

  // Clear placeholder, render user message
  const ph = document.querySelector("#messages .placeholder");
  if (ph) ph.remove();
  addMessage("user", q);
  $("chat-input").value = "";
  $("chat-input").disabled = true;
  $("send-btn").disabled = true;

  // Thinking indicator
  const thinking = addMessage("thinking",
    "PaperMind is thinking... local inference, this takes a few minutes");

  try {
    const r = await fetch(API + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    const data = await r.json();
    thinking.remove();
    addMessage("assistant", data.answer);
    renderSources(data.answer);
    renderTrace(data.trace || []);
  } catch (err) {
    thinking.remove();
    addMessage("assistant", "Error: " + err.message);
  } finally {
    $("chat-input").disabled = false;
    $("send-btn").disabled = false;
    $("chat-input").focus();
  }
});

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  if (role === "thinking") {
    div.innerHTML = "<span class='pulse'></span>" + text;
  } else {
    div.textContent = text;
  }
  $("messages").appendChild(div);
  $("messages").scrollTop = $("messages").scrollHeight;
  return div;
}

// ---------- Sources panel ----------
function renderSources(answer) {
  // Extract [chunk_id] citations from the answer text
  const ids = [...new Set((answer.match(/\[([^\]]+)\]/g) || [])
    .map((m) => m.slice(1, -1))
    .filter((s) => s.includes("_c")))];
  if (!ids.length) {
    $("sources-list").innerHTML = "<em>No citations in this answer.</em>";
    return;
  }
  $("sources-list").innerHTML = ids.map((id) =>
    "<div class='cite' data-id='" + id + "'>" + id + "</div>").join("");
  document.querySelectorAll(".cite").forEach((el) => {
    el.addEventListener("click", () => toggleCitation(el));
  });
}

async function toggleCitation(el) {
  const existing = el.querySelector(".cite-text");
  if (existing) { existing.remove(); return; }
  const id = el.dataset.id;
  el.insertAdjacentHTML("beforeend",
    "<div class='cite-text'>loading...</div>");
  try {
    const r = await fetch(API + "/cite/" + encodeURIComponent(id));
    const data = await r.json();
    el.querySelector(".cite-text").textContent =
      data.text || "(not found)";
  } catch {
    el.querySelector(".cite-text").textContent = "(error)";
  }
}

// ---------- Trace panel ----------
function renderTrace(trace) {
  if (!trace.length) {
    $("tools-list").innerHTML = "<em>No tool calls.</em>";
    return;
  }
  $("tools-list").innerHTML = trace.map((t) => {
    const arg = JSON.stringify(t.args);
    return "<div class='tool-call'>" + t.tool + "<br><span style='opacity:.6'>"
      + arg + "</span></div>";
  }).join("");
}

// ---------- KG viewer (static top-degree overview) ----------
async function loadKG() {
  try {
    const r = await fetch(API + "/graph");
    const data = await r.json();
    if (!data.nodes || !data.nodes.length) {
      $("kg-caption").textContent = "graph empty";
      return;
    }
    const nodes = new vis.DataSet(data.nodes);
    const edges = new vis.DataSet(data.edges);
    new vis.Network($("kg-viewer"), { nodes, edges }, {
      nodes: { shape: "dot", size: 10, font: { size: 11 } },
      edges: { color: { opacity: 0.4 }, smooth: false },
      physics: { stabilization: true, barnesHut: { gravitationalConstant: -3000 } },
      interaction: { hover: true },
    });
    $("kg-caption").textContent =
      data.nodes.length + " concepts shown (top-degree) of " + data.total + " total";
  } catch {
    $("kg-caption").textContent = "graph unavailable";
  }
}

// ---------- Init ----------
refreshLibrary();
loadKG();
