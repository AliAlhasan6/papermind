"""Knowledge graph: schemas, LLM extraction prompt, NetworkX helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import pickle

import networkx as nx
from pydantic import BaseModel, Field, field_validator


# ---------- Schemas ----------

EntityType = Literal["concept", "method", "dataset", "metric", "author", "system", "other"]
RelationType = Literal["uses", "extends", "compares", "cites", "part_of", "evaluates_on", "other"]


class Entity(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    type: EntityType = "other"

    @field_validator("name")
    @classmethod
    def normalize(cls, v: str) -> str:
        # Normalize whitespace; preserve case (multilingual content)
        return " ".join(v.strip().split())


class Relation(BaseModel):
    src: str = Field(min_length=2, max_length=120)
    dst: str = Field(min_length=2, max_length=120)
    type: RelationType = "other"

    @field_validator("src", "dst")
    @classmethod
    def normalize(cls, v: str) -> str:
        return " ".join(v.strip().split())


class ChunkKG(BaseModel):
    """What the LLM is asked to return per chunk."""
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


# ---------- Persistence ----------

def load_graph(path: Path) -> nx.DiGraph:
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    return nx.DiGraph()


def save_graph(graph: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(graph, f)


def merge_chunk_kg(graph: nx.DiGraph, kg: ChunkKG, chunk_id: str) -> tuple[int, int]:
    """Merge per-chunk KG into the global graph. Returns (new_nodes, new_edges)."""
    new_nodes = 0
    new_edges = 0

    for e in kg.entities:
        if e.name not in graph:
            graph.add_node(e.name, type=e.type, src_chunks=[chunk_id])
            new_nodes += 1
        else:
            srcs = graph.nodes[e.name].setdefault("src_chunks", [])
            if chunk_id not in srcs:
                srcs.append(chunk_id)

    for r in kg.relations:
        # Auto-add endpoint nodes if extraction missed them
        for node in (r.src, r.dst):
            if node not in graph:
                graph.add_node(node, type="other", src_chunks=[chunk_id])
                new_nodes += 1
        if not graph.has_edge(r.src, r.dst):
            graph.add_edge(r.src, r.dst, type=r.type, src_chunks=[chunk_id])
            new_edges += 1
        else:
            srcs = graph[r.src][r.dst].setdefault("src_chunks", [])
            if chunk_id not in srcs:
                srcs.append(chunk_id)

    return new_nodes, new_edges
