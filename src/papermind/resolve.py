"""Entity resolution: merge duplicate nodes differing only by case/whitespace/plural.

String-level only. Semantic entity merging is intentionally out of scope
(documented as future work).

Usage:
    python -m papermind.resolve
"""
from __future__ import annotations
import pickle
import shutil
from pathlib import Path

import networkx as nx

from papermind.config import KG_PATH


def canonical(name: str) -> str:
    """Map an entity name to a canonical merge key."""
    k = " ".join(name.strip().split()).lower()
    if k.isascii() and len(k) > 3 and k.endswith("s") and not k.endswith("ss"):
        k = k[:-1]
    return k


def resolve(graph: nx.DiGraph) -> nx.DiGraph:
    """Return a new graph with case/plural-duplicate nodes merged."""
    groups: dict[str, list[str]] = {}
    for node in graph.nodes():
        groups.setdefault(canonical(node), []).append(node)

    canon_to_display: dict[str, str] = {}
    for key, members in groups.items():
        best = max(members, key=lambda n: (graph.degree(n), -len(n)))
        canon_to_display[key] = best

    new = nx.DiGraph()
    for key, members in groups.items():
        display = canon_to_display[key]
        all_chunks: list[str] = []
        types: list[str] = []
        for m in members:
            all_chunks += graph.nodes[m].get("src_chunks", [])
            types.append(graph.nodes[m].get("type", "other"))
        non_other = [t for t in types if t != "other"]
        node_type = max(set(non_other), key=non_other.count) if non_other else "other"
        new.add_node(display, type=node_type, src_chunks=sorted(set(all_chunks)))

    for u, v, attrs in graph.edges(data=True):
        nu = canon_to_display[canonical(u)]
        nv = canon_to_display[canonical(v)]
        if nu == nv:
            continue
        if new.has_edge(nu, nv):
            existing = new[nu][nv].setdefault("src_chunks", [])
            for c in attrs.get("src_chunks", []):
                if c not in existing:
                    existing.append(c)
        else:
            new.add_edge(nu, nv, **attrs)
    return new


def main():
    path = Path(KG_PATH)
    graph = pickle.load(path.open("rb"))
    before_n, before_e = graph.number_of_nodes(), graph.number_of_edges()
    before_comp = nx.number_connected_components(graph.to_undirected())

    resolved = resolve(graph)
    after_n, after_e = resolved.number_of_nodes(), resolved.number_of_edges()
    comps = sorted(nx.connected_components(resolved.to_undirected()), key=len, reverse=True)

    shutil.copy(path, path.with_suffix(".pkl.bak"))
    pickle.dump(resolved, path.open("wb"))

    print(f"Nodes:      {before_n} -> {after_n}  ({before_n - after_n} merged)")
    print(f"Edges:      {before_e} -> {after_e}")
    print(f"Components: {before_comp} -> {len(comps)}")
    print(f"Largest:    {len(comps[0])} nodes ({100*len(comps[0])/after_n:.0f}% of graph)")
    print(f"Backup: {path.with_suffix('.pkl.bak')}")


if __name__ == "__main__":
    main()
