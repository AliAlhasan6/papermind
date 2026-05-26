"""The three agent tools: vector_search, graph_neighbors, cite_source.

Each is a LangChain @tool — the agent (Day 5) binds these and decides which to call.
All three are also plain-callable for standalone testing.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # embeddings on CPU

import pickle
from functools import lru_cache
from pathlib import Path

import chromadb
import networkx as nx
from chromadb.config import Settings
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

from papermind.config import (
    CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL, KG_PATH,
)


# ---------- Lazy-loaded singletons ----------

@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL, device="cpu")


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# Cached for the server's lifetime by design; restart the server to pick up a rebuilt KG.
@lru_cache(maxsize=1)
def _graph() -> nx.DiGraph:
    path = Path(KG_PATH)
    if not path.exists():
        return nx.DiGraph()
    with path.open("rb") as f:
        return pickle.load(f)


# ---------- Tool 1: vector search ----------

@tool
def vector_search(query: str, k: int = 5) -> str:
    """Semantic search over the paper corpus.

    Use this for general questions about what the papers say. Returns the top-k
    most relevant text chunks, each tagged with its chunk ID in [brackets].
    """
    col = _collection()
    emb = _embedder().encode([query]).tolist()
    res = col.query(query_embeddings=emb, n_results=k)

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    if not ids:
        return "No results found in the corpus."

    blocks = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        src = meta.get("source", "unknown")
        sim = 1 - dist  # cosine distance -> rough similarity
        blocks.append(f"[{cid}] (from {src}, relevance {sim:.2f})\n{doc.strip()}")
    return "\n\n---\n\n".join(blocks)


# ---------- Tool 2: graph neighbors ----------

@tool
def graph_neighbors(entity: str, hops: int = 1) -> str:
    """Find concepts related to a given entity in the knowledge graph.

    Use this for questions about RELATIONSHIPS between concepts (what extends what,
    what uses what). Returns neighboring entities, the relation types, and the
    chunk IDs where each relationship was found. If the entity is not found, try
    vector_search first to discover the exact entity name.
    """
    g = _graph()
    if g.number_of_nodes() == 0:
        return "Knowledge graph is empty."

    # Exact match first, then case-insensitive fallback
    match = entity if entity in g else None
    if match is None:
        low = entity.strip().lower()
        for n in g.nodes():
            if n.lower() == low:
                match = n
                break
    if match is None:
        # Substring suggestion
        low = entity.strip().lower()
        hits = [n for n in g.nodes() if low in n.lower()][:8]
        if hits:
            return (f"Entity '{entity}' not found exactly. Similar entities: "
                    + ", ".join(hits)
                    + ". Retry graph_neighbors with one of these, or use vector_search.")
        return f"Entity '{entity}' not found in the knowledge graph. Use vector_search instead."

    # Collect neighbors within `hops`, via the undirected view
    ug = g.to_undirected()
    dist = nx.single_source_shortest_path_length(ug, match, cutoff=hops)

    lines = [f"Entity: {match} (type: {g.nodes[match].get('type', 'other')})"]
    src_chunks = g.nodes[match].get("src_chunks", [])
    if src_chunks:
        lines.append(f"Appears in chunks: {', '.join(src_chunks[:5])}")

    neighbors = [(n, d) for n, d in dist.items() if n != match]
    if not neighbors:
        lines.append("No connected entities (this concept is isolated in the graph).")
        return "\n".join(lines)

    lines.append(f"\nRelated entities ({len(neighbors)} within {hops} hop(s)):")
    for n, d in sorted(neighbors, key=lambda x: x[1])[:15]:
        # Find the relation type + source chunk on the connecting edge
        edge = g.get_edge_data(match, n) or g.get_edge_data(n, match) or {}
        rel = edge.get("type", "related")
        chunks = edge.get("src_chunks", [])
        chunk_hint = f" [{chunks[0]}]" if chunks else ""
        lines.append(f"  - {n}  ({rel}, {d} hop){chunk_hint}")
    return "\n".join(lines)


# ---------- Tool 3: cite source ----------

@tool
def cite_source(chunk_id: str) -> str:
    """Fetch the exact verbatim text of a chunk by its ID.

    Use this to verify a claim or quote precisely before giving a final answer.
    Pass a chunk ID like 'paper_c42' (the IDs returned by the other two tools).
    """
    col = _collection()
    res = col.get(ids=[chunk_id.strip()])
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    if not docs:
        return f"Chunk '{chunk_id}' not found."
    src = metas[0].get("source", "unknown") if metas else "unknown"
    return f"[{chunk_id}] from {src}:\n\n{docs[0].strip()}"


# All tools, for the agent to bind
ALL_TOOLS = [vector_search, graph_neighbors, cite_source]
