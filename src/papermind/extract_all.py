"""Resumable overnight KG extraction over the full Chroma corpus.

Usage:
    python -m papermind.extract_all                 # process all chunks
    python -m papermind.extract_all --limit 20      # smoke test (20 chunks)
    python -m papermind.extract_all --resume        # skip already-done (default ON)

Outputs:
    kg.pkl                       — the NetworkX graph
    data/kg_progress.jsonl       — one line per processed chunk (for resume)
    data/kg_failures.jsonl       — one line per chunk that failed extraction
"""
from __future__ import annotations
import argparse
import json
import signal
import sys
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings

from papermind.config import CHROMA_PATH, COLLECTION_NAME, KG_PATH
from papermind.extract import extract_chunk
from papermind.kg import load_graph, save_graph, merge_chunk_kg


PROGRESS_LOG = Path("data/kg_progress.jsonl")
FAILURES_LOG = Path("data/kg_failures.jsonl")
SAVE_EVERY = 10  # checkpoint graph every N chunks


def _load_done() -> set[str]:
    """Read progress log, return set of chunk IDs already processed."""
    if not PROGRESS_LOG.exists():
        return set()
    done = set()
    with PROGRESS_LOG.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["chunk_id"])
            except Exception:
                pass
    return done


def _append_progress(chunk_id: str, n_ent: int, n_rel: int, seconds: float) -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_LOG.open("a") as f:
        f.write(json.dumps({
            "chunk_id": chunk_id,
            "entities": n_ent,
            "relations": n_rel,
            "seconds": round(seconds, 1),
        }) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only N chunks (smoke test)")
    parser.add_argument("--no-resume", action="store_true", help="Re-process everything from scratch")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    args = parser.parse_args()

    # Open Chroma
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(COLLECTION_NAME)
    all_data = col.get()  # pull all IDs + documents
    chunk_ids = all_data["ids"]
    chunk_docs = all_data["documents"]
    total = len(chunk_ids)

    # Resume logic
    done = set() if args.no_resume else _load_done()
    todo = [(cid, doc) for cid, doc in zip(chunk_ids, chunk_docs) if cid not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"Corpus: {total} chunks · already done: {len(done)} · to process: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    # Load existing graph
    graph = load_graph(Path(KG_PATH))
    print(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Graceful interrupt: save graph and exit on Ctrl+C
    def _on_interrupt(signum, frame):
        print("\n\nInterrupt received — saving graph before exit…")
        save_graph(graph, Path(KG_PATH))
        print(f"Saved: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        sys.exit(0)
    signal.signal(signal.SIGINT, _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)

    # Main loop
    t_start = time.perf_counter()
    for i, (cid, doc) in enumerate(todo, start=1):
        t0 = time.perf_counter()
        kg = extract_chunk(doc, cid, failures_log=FAILURES_LOG)
        new_n, new_e = merge_chunk_kg(graph, kg, cid)
        dt = time.perf_counter() - t0
        _append_progress(cid, len(kg.entities), len(kg.relations), dt)

        # Progress line: where we are, what just happened, ETA
        avg = (time.perf_counter() - t_start) / i
        eta_min = (len(todo) - i) * avg / 60
        print(f"[{i:4d}/{len(todo)}] {cid[:50]:50s}  "
              f"+{len(kg.entities)}e/{len(kg.relations)}r  ({dt:.1f}s)  "
              f"graph: {graph.number_of_nodes()}n/{graph.number_of_edges()}e  "
              f"ETA: {eta_min:.0f}m")

        # Checkpoint
        if i % args.save_every == 0:
            save_graph(graph, Path(KG_PATH))

    # Final save
    save_graph(graph, Path(KG_PATH))
    elapsed = (time.perf_counter() - t_start) / 60
    print(f"\nDone: {len(todo)} chunks in {elapsed:.1f}m")
    print(f"Final graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


if __name__ == "__main__":
    main()
