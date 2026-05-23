"""FastAPI app for PaperMind.

Endpoints:
  POST /ingest   — upload PDFs, run vector ingestion
  GET  /papers   — list papers in the corpus
  POST /chat     — ask the agent a question (slow: local inference)
  GET  /health   — liveness check
  GET  /         — serve the frontend
"""
from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
from collections import Counter

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from papermind.config import CHROMA_PATH, COLLECTION_NAME
from papermind.ingest import ingest_folder

app = FastAPI(title="PaperMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Ingestion ----------

@app.post("/ingest")
async def ingest(files: list[UploadFile]):
    """Receive uploaded PDFs, write to a temp dir, run ingestion."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for f in files:
            dest = tmp_path / f.filename
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
        summary = ingest_folder(tmp_path)
    return summary


@app.get("/papers")
def papers():
    """List papers currently in the Chroma collection."""
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        return {"papers": []}
    all_meta = col.get()["metadatas"] or []
    counts = Counter(m["paper_id"] for m in all_meta)
    return {"papers": [{"name": pid, "chunks": n} for pid, n in sorted(counts.items())]}


# ---------- Chat ----------

class ChatRequest(BaseModel):
    question: str
    thread_id: str = "default"


@app.post("/chat")
def chat(req: ChatRequest):
    """Ask PaperMind a question. Returns answer + tool trace.

    Note: local inference is slow (~minutes per query on CPU/low-end GPU).
    Imported lazily so the server boots fast and ingestion works without Ollama.
    """
    from papermind.agent import ask
    return ask(req.question, thread_id=req.thread_id)


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Cite + Graph endpoints (Day 6) ----------
@app.get("/cite/{chunk_id}")
def cite(chunk_id: str):
    """Return verbatim text of a chunk, for the Sources panel."""
    from papermind.tools import cite_source
    return {"chunk_id": chunk_id, "text": cite_source.invoke({"chunk_id": chunk_id})}


@app.get("/graph")
def graph(top: int = 40):
    """Return the top-degree subgraph for the KG viewer."""
    import pickle
    from papermind.config import KG_PATH
    path = Path(KG_PATH)
    if not path.exists():
        return {"nodes": [], "edges": [], "total": 0}
    g = pickle.load(path.open("rb"))
    top_nodes = [n for n, _ in sorted(g.degree(), key=lambda x: -x[1])[:top]]
    keep = set(top_nodes)
    nodes = [{"id": n, "label": n[:24], "value": g.degree(n)} for n in top_nodes]
    edges = [{"from": u, "to": v} for u, v in g.edges() if u in keep and v in keep]
    return {"nodes": nodes, "edges": edges, "total": g.number_of_nodes()}


# ---------- Serve the frontend ----------

frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    def root():
        return FileResponse(str(frontend_dir / "index.html"))

    @app.get("/style.css")
    def style():
        return FileResponse(str(frontend_dir / "style.css"))

    @app.get("/app.js")
    def js():
        return FileResponse(str(frontend_dir / "app.js"))
