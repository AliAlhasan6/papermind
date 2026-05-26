"""LLM-driven entity + relation extraction.

Calls Ollama (Qwen 2.5) with strict JSON output. Validates with Pydantic.
Logs failures to data/kg_failures.jsonl for inspection.
"""
from __future__ import annotations
import json
from pathlib import Path

from langchain_ollama import ChatOllama

from papermind.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from papermind.kg import ChunkKG


EXTRACT_PROMPT = """You extract a tiny knowledge graph from a research paper excerpt.

Output ONLY valid minified JSON matching this exact shape:
{{"entities": [{{"name": "...", "type": "concept|method|dataset|metric|author|system|other"}}],
 "relations": [{{"src": "...", "dst": "...", "type": "uses|extends|compares|cites|part_of|evaluates_on|other"}}]}}

Rules:
- 0 to 8 entities, 0 to 6 relations. Quality over quantity.
- Use the language of the excerpt for entity names (do not translate).
- Skip generic words ("paper", "section", "Figure 1"). Skip pronouns.
- Only add a relation if BOTH src and dst are in your entities list.
- Empty arrays are fine if the excerpt is uninformative (acknowledgments, page numbers, etc).
- Return ONLY the JSON. No prose, no markdown, no code fences.

Excerpt:
{chunk}

JSON:"""


def _strip_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if "```" in t[3:] else t[3:]
        t = t.removeprefix("json").strip()
        t = t.rsplit("```", 1)[0].strip() if "```" in t else t
    return t


_llm_cache: ChatOllama | None = None


def _llm() -> ChatOllama:
    """Lazy-init the LLM (avoids loading at import time)."""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            format="json",  # Ollama's native JSON mode
            num_predict=400,  # cap output tokens so it can't ramble
        )
    return _llm_cache


def extract_chunk(chunk_text: str, chunk_id: str, failures_log: Path | None = None) -> ChunkKG:
    """Extract entities + relations from a single chunk. Returns empty KG on failure."""
    prompt = EXTRACT_PROMPT.format(chunk=chunk_text[:1500])  # cap input too
    raw = None
    try:
        raw = _llm().invoke(prompt).content
        cleaned = _strip_fences(str(raw))
        data = json.loads(cleaned)
        kg = ChunkKG.model_validate(data)
        # Cap runaway extractions (reference lists, TOCs dump too many entities)
        if len(kg.entities) > 8:
            kg.entities = kg.entities[:8]
        if len(kg.relations) > 6:
            kg.relations = kg.relations[:6]
        return kg
    except Exception as e:
        if failures_log is not None:
            failures_log.parent.mkdir(parents=True, exist_ok=True)
            with failures_log.open("a") as f:
                f.write(json.dumps({
                    "chunk_id": chunk_id,
                    "error": str(e)[:200],
                    "raw": str(raw)[:500] if raw is not None else None,
                }) + "\n")
        return ChunkKG()
