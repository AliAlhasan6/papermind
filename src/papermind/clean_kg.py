"""Drop obvious noise nodes from the KG: math symbols, OCR fragments.

Conservative: keeps short pure-letter acronyms (ИВ, ВУ, ФУ are real concepts).
Only removes names containing math unicode, or that are mostly digits/symbols.

Usage:
    python -m papermind.clean_kg
"""
from __future__ import annotations
import pickle, shutil, re
from pathlib import Path
import networkx as nx
from papermind.config import KG_PATH

# Unicode math / italic-math / operator ranges -> formula fragment, not concept
_MATH = re.compile(r"[\U0001D400-\U0001D7FF\u2200-\u22FF\u2100-\u214F]")

def is_noise(name: str) -> bool:
    n = name.strip()
    if not n:
        return True
    # Any math/formula unicode -> noise
    if _MATH.search(n):
        return True
    # Single character -> noise (too granular to be a concept)
    if len(n) == 1:
        return True
    # Mostly non-letters (digits, punctuation, symbols) -> noise.
    # NOTE: short pure-letter acronyms (ИВ, ВУ) pass this and are KEPT.
    alpha = sum(c.isalpha() for c in n)
    if alpha / len(n) < 0.5:
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
    print(f"Sample removed: {junk[:15]}")
    print(f"Backup: {path.with_suffix('.pkl.preclean')}")

if __name__ == "__main__":
    main()
