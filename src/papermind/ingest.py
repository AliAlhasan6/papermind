"""PDF → chunks (Chroma).

Day 2: vector ingestion only. KG extraction added Day 3.

Usage:
    python -m papermind.ingest data/papers/
    python -m papermind.ingest /any/folder/with/pdfs
"""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only for embeddings (Pascal GPU not supported by torch 2.12)


import sys
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from papermind.config import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _extract_text(pdf_path: Path) -> str:
    """Read all pages of a PDF into a single string."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"  ⚠ page extraction failed: {e}")
            text = ""
        pages.append(text)
    return "\n\n".join(pages)


def _get_collection():
    """Get or create the persistent Chroma collection."""
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_pdf(pdf_path: Path, embedder: SentenceTransformer, collection) -> int:
    """Ingest a single PDF. Returns number of chunks added."""
    paper_id = pdf_path.stem.replace(" ", "_")
    print(f"→ {pdf_path.name}")

    text = _extract_text(pdf_path)
    if not text.strip():
        print(f"  ⚠ no text extracted, skipping")
        return 0

    chunks = _splitter().split_text(text)
    if not chunks:
        print(f"  ⚠ no chunks produced, skipping")
        return 0

    # Compute embeddings (this is the slow step on CPU; ~1s per 10 chunks)
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

    ids = [f"{paper_id}_c{i}" for i in range(len(chunks))]
    metadatas = [
        {"paper_id": paper_id, "chunk_idx": i, "source": pdf_path.name}
        for i in range(len(chunks))
    ]

    # Upsert (replaces if same ID exists — safe to re-run)
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"  ✓ {len(chunks)} chunks → Chroma")
    return len(chunks)


def ingest_folder(folder: Path) -> dict:
    """Ingest every PDF in a folder. Returns summary stats."""
    pdfs = sorted(folder.glob("*.pdf")) + sorted(folder.glob("*.PDF"))
    if not pdfs:
        print(f"No PDFs found in {folder}")
        return {"papers": 0, "chunks": 0, "seconds": 0.0}

    print(f"Loading embedder: {EMBED_MODEL} (first run downloads ~80 MB)")
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

    print(f"Opening Chroma collection at {CHROMA_PATH}")
    collection = _get_collection()

    t0 = time.perf_counter()
    total_chunks = 0
    for pdf in pdfs:
        total_chunks += ingest_pdf(pdf, embedder, collection)
    dt = time.perf_counter() - t0

    summary = {"papers": len(pdfs), "chunks": total_chunks, "seconds": round(dt, 1)}
    print(f"\nDone: {summary['papers']} papers, {summary['chunks']} chunks, {summary['seconds']}s")
    print(f"Collection size: {collection.count()} total chunks")
    return summary


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/papers")
    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)
    ingest_folder(folder)
