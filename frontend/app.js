// PaperMind frontend — Atlas design, real backend wiring.
const API = "";
const $ = (id) => document.getElementById(id);
let pendingFiles = [];

// ---------- Ingestion ----------
$("pdf-input").addEventListener("change", (e) => {
  pendingFiles = Array.from(e.target.files);
  const f = $("file-field-text-holder");
  if (f) f.textContent = pendingFiles.length
    ? pendingFiles.length + " file(s) selected" : "Choose PDFs...";
  $("ingest-status").textContent = pendingFiles.length
    ? pendingFiles.length + " file(s) ready" : "";
});

$("ingest-btn").addEventListener("click", async () => {
  if (!pendingFiles.length) { $("ingest-status").textContent = "no files selected"; return; }
  $("ingest-btn").disabled = true;
  $("ingest-status").textContent = "ingesting... (may take a minute)";
  const fd = new FormData();
  pendingFiles.forEach((f) => fd.append("files", f));
  try {
    const r = await fetch(API + "/ingest", { method: "POST", body: fd });
    const data = await r.json();
    $("ingest-status").textContent =
      data.papers + " papers indexed, " + data.chunks + " chunks";
    refreshLibrary();
    loadKG();
    pendingFiles = [];
    $("pdf-input").value = "";
  } catch (err) {
    $("ingest-status").textContent = "error: " + err.message;
  } finally {
    $("ingest-btn").disabled = false;
  }
});

async function refreshLibrary() {
  try {
    const r = await fetch(API + "/papers");
    const data = await r.json();
    const list = $("paper-list");
    if (!data.papers || !data.papers.length) {
      list.innerHTML = '<div class="empty"><p class="empty-title">No papers loaded</p>'
        + '<p class="empty-body">Drop PDFs above. PaperMind will not answer '
        + 'from anything outside this library.</p></div>';
      return;
    }
    list.innerHTML = data.papers.map((p) =>
      '<div class="paper" data-id="' + escapeAttr(p.name) + '">'
      + '<div style="display:flex;justify-content:space-between;gap:8px;align-items:baseline">'
      + '<div class="title">' + escapeHtml(p.name) + '</div>'
      + '<div style="display:flex;gap:8px;align-items:baseline">'
      + '<span class="lang">' + p.chunks + ' ch</span>'
      + '<button class="paper-del" data-id="' + escapeAttr(p.name)
      + '" title="Remove this paper" aria-label="Remove">&times;</button>'
      + '</div></div></div>'
    ).join("");
    list.querySelectorAll(".paper-del").forEach((b) => {
      b.addEventListener("click", (e) => { e.stopPropagation(); deletePaper(b.dataset.id); });
    });
    const count = $("paper-count");
    if (count) count.textContent = data.papers.length;
  } catch {
    $("paper-list").innerHTML =
      '<div class="empty"><p class="empty-body">API offline.</p></div>';
  }
}

// ---------- Chat ----------
$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("chat-input").value.trim();
  if (!q) return;

  const welcome = document.querySelector("#messages .welcome");
  if (welcome) welcome.remove();

  addMessage("user", q);
  $("chat-input").value = "";
  $("chat-input").disabled = true;
  $("send-btn").disabled = true;

  const thinking = document.createElement("div");
  thinking.className = "thinking";
  thinking.textContent = "PaperMind is thinking — local inference, a few minutes";
  $("messages").appendChild(thinking);
  $("messages").scrollTop = $("messages").scrollHeight;

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
  div.className = "message " + role;
  if (role === "assistant") {
    // Split into paragraphs, turn [chunk_id] into citation chips
    text.split(/\n\n+/).forEach((para) => {
      const p = document.createElement("p");
      p.innerHTML = escapeHtml(para).replace(/\[([^\]]+_c\d+)\]/g,
        '<span class="cite" data-cite="$1">$1</span>');
      div.appendChild(p);
    });
  } else {
    div.textContent = text;
  }
  $("messages").appendChild(div);
  $("messages").scrollTop = $("messages").scrollHeight;
  return div;
}

// ---------- Sources ----------
function renderSources(answer) {
  const ids = [...new Set((answer.match(/\[([^\]]+_c\d+)\]/g) || [])
    .map((m) => m.slice(1, -1)))];
  const panel = $("sources-list");
  if (!ids.length) {
    panel.innerHTML =
      '<div class="empty"><p class="empty-body">No citations in this answer.</p></div>';
    return;
  }
  panel.innerHTML = ids.map((id) =>
    '<div class="source" aria-expanded="false" data-id="' + escapeAttr(id) + '">'
    + '<div class="source-header">'
    + '<span class="cit">&sect;</span>'
    + '<span class="paper">' + escapeHtml(id) + '</span></div>'
    + '<div class="source-body">click to load source text...</div></div>'
  ).join("");
  panel.querySelectorAll(".source").forEach((el) => {
    el.addEventListener("click", () => toggleSource(el));
  });
}

async function toggleSource(el) {
  const open = el.getAttribute("aria-expanded") === "true";
  el.setAttribute("aria-expanded", open ? "false" : "true");
  el.classList.toggle("expanded", !open);
  if (open) return;
  const body = el.querySelector(".source-body");
  if (body.dataset.loaded) return;
  try {
    const r = await fetch(API + "/cite/" + encodeURIComponent(el.dataset.id));
    const data = await r.json();
    body.textContent = data.text || "(not found)";
    body.dataset.loaded = "1";
  } catch {
    body.textContent = "(error loading source)";
  }
}

// ---------- Trace ----------
function renderTrace(trace) {
  const panel = $("tools-list");
  if (!trace.length) {
    panel.innerHTML =
      '<div class="empty"><p class="empty-body">No tool calls.</p></div>';
    return;
  }
  panel.innerHTML = trace.map((t) => {
    const args = escapeHtml(JSON.stringify(t.args));
    return '<div class="tool" data-tool="' + escapeAttr(t.tool) + '">'
      + '<span></span>'
      + '<div><div><span class="tool-name">' + escapeHtml(t.tool) + '</span>'
      + '<span class="tool-args"> ' + args + '</span></div></div>'
      + '<span class="tool-time"></span></div>';
  }).join("");
}

// ---------- KG viewer ----------
async function loadKG() {
  try {
    const r = await fetch(API + "/graph");
    const data = await r.json();
    const cap = $("kg-caption");
    if (!data.nodes || !data.nodes.length) {
      if (cap) cap.innerHTML = '<span class="kg-caption-empty">graph empty</span>';
      return;
    }
    const nodes = new vis.DataSet(data.nodes.map((n) => ({
      id: n.id, label: n.label, value: n.value,
    })));
    const edges = new vis.DataSet(data.edges.map((e) => ({
      from: e.from, to: e.to,
    })));
    const kgNet = new vis.Network($("kg-viewer"), { nodes, edges }, {
      nodes: { shape: "dot", size: 9, font: { size: 11, face: "IBM Plex Mono" } },
      edges: { color: { opacity: 0.35 }, smooth: false },
      physics: { stabilization: { iterations: 200 },
                 barnesHut: { gravitationalConstant: -3000 } },
      interaction: { hover: true },
    });
    kgNet.on("click", async (params) => {
      if (!params.nodes.length) return;
      const nodeName = params.nodes[0];
      const box = $("kg-caption");
      box.innerHTML = "<span class='kg-caption-empty'>loading...</span>";
      try {
        const r = await fetch(API + "/node/" + encodeURIComponent(nodeName));
        const d = await r.json();
        if (!d.found) {
          box.innerHTML = "<span class='kg-caption-empty'>node not found</span>";
          return;
        }
        const nbrs = d.neighbors.length
          ? d.neighbors.map((n) =>
              "<div class='node-nbr'>" + escapeHtml(n.relation)
              + " &rarr; " + escapeHtml(n.name) + "</div>").join("")
          : "<div class='kg-caption-empty'>no connected concepts</div>";
        box.innerHTML =
          "<div class='node-info'>"
          + "<div class='node-title'>" + escapeHtml(d.name) + "</div>"
          + "<div class='node-meta'>type: " + escapeHtml(d.type)
          + " &middot; degree: " + d.degree + "</div>"
          + "<div class='node-nbrs'>" + nbrs + "</div>"
          + "</div>";
      } catch (err) {
        box.innerHTML = "<span class='kg-caption-empty'>error: "
          + escapeHtml(err.message) + "</span>";
      }
    });

    if (cap) cap.textContent =
      data.nodes.length + " concepts shown of " + data.total + " total";
  } catch {
    const cap = $("kg-caption");
    if (cap) cap.innerHTML = '<span class="kg-caption-empty">graph unavailable</span>';
  }
}

// ---------- Delete a paper ----------
async function deletePaper(paperId) {
  if (!confirm('Remove "' + paperId + '" from the library?\n'
      + "This deletes its chunks and graph nodes.")) return;
  try {
    const r = await fetch(API + "/papers/" + encodeURIComponent(paperId),
                          { method: "DELETE" });
    const data = await r.json();
    if (data.error) { alert("Delete failed: " + data.error); return; }
    refreshLibrary();
    loadKG();
  } catch (err) {
    alert("Delete failed: " + err.message);
  }
}

// ---------- Helpers ----------
function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function escapeAttr(s) { return escapeHtml(s).replace(/'/g, "&#39;"); }

// ---------- Init ----------
refreshLibrary();
loadKG();

// ---------- Theme toggle ----------
(function () {
  const root = document.documentElement;
  const saved = localStorage.getItem("papermind-theme");
  if (saved === "dark" || saved === "light") {
    root.setAttribute("data-theme", saved);
  }
  const btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.addEventListener("click", () => {
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      const current = root.getAttribute("data-theme")
        || (prefersDark ? "dark" : "light");
      const next = current === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      localStorage.setItem("papermind-theme", next);
    });
  }
})();
