"""Drop obvious noise nodes from the KG: math symbols, OCR fragments, stubs.

Run after resolve.py. Conservative — only removes clearly-junk nodes.

Usage:
    python -m papermind.clean_kg
"""
from __future__ import annotations
import pickle, shutil, re
from pathlib import Path
import networkx as nx
from papermind.config import KG_PATH

# Unicode math/italic ranges that signal a formula fragment, not a concept
_MATH = re.compile(r"[\U0001D400-\U0001D7FF\u2200-\u22FF\u2100-\u214F]")

def is_noise(name: str) -> bool:
    n = name.strip()
    if len(n) < 3:
        return True
    if _MATH.search(n):
        return True
    # mostly digits / punctuation
    alpha = sum(c.isalpha() for c in n)
    if alpha / max(len(n), 1) < 0.5:
        return True
    return False

def main():
    path = Path(KG_PATH)
    g = pickle.load(path.open("rb"))
    before = g.number_of_nodes()
    junk = [n for n in g.nodes() if is_noise(n)]
    g.remove_nodes_from(junk)
    shutil.copy(path, path.with_suffix(".pkl.preclean"))
    pickle.dump(g, path.open("wb"))
    print(f"Removed {len(junk)} noise nodes ({before} -> {g.number_of_nodes()})")
    print(f"Edges now: {g.number_of_edges()}")
    print(f"Sample removed: {junk[:12]}")
    print(f"Backup: {path.with_suffix('.pkl.preclean')}")

if __name__ == "__main__":
    main()
