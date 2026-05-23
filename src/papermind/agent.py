"""The PaperMind LangGraph agent.

A ReAct agent over three tools (vector_search, graph_neighbors, cite_source)
backed by a local Ollama LLM. Streams or returns a cited answer.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from functools import lru_cache

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from papermind.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from papermind.tools import ALL_TOOLS


SYSTEM_PROMPT = """You are PaperMind, a research assistant that answers questions \
strictly from an indexed corpus of research papers.

You have three tools:
- vector_search: semantic search for what the papers SAY about a topic.
- graph_neighbors: find concepts RELATED to an entity (relationships, what extends/uses what).
- cite_source: fetch the exact text of a chunk by its ID, to verify before answering.

Rules:
1. ALWAYS ground your answer in retrieved chunks. Never answer from prior knowledge.
2. Cite chunk IDs in square brackets, e.g. [viksnin-i.i.-dissertacija_c137], for every claim.
3. For "what does X say" questions, start with vector_search.
4. For "what relates to X" / "what extends X" questions, use graph_neighbors.
5. If a tool returns nothing useful, try another tool or a reworded query.
6. If the corpus genuinely does not contain the answer, say so plainly. Do not invent.
7. Keep answers concise and factual. The corpus may be multilingual; answer in the \
   user's language but keep cited terms in their original language.
"""


@lru_cache(maxsize=1)
def get_agent():
    """Build (once) and return the compiled LangGraph agent."""
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),  # in-memory conversation state
    )
    return agent


def ask(question: str, thread_id: str = "default") -> dict:
    """Ask PaperMind a question. Returns the final answer + the tool trace."""
    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )

    messages = result["messages"]
    final_answer = messages[-1].content

    # Reconstruct the tool trace for the UI's "Agent trace" panel
    trace = []
    for m in messages:
        for tc in getattr(m, "tool_calls", []) or []:
            trace.append({"tool": tc["name"], "args": tc.get("args", {})})

    return {"answer": final_answer, "trace": trace, "n_steps": len(messages)}


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is the main goal of this dissertation?"
    print(f"Q: {q}\n")
    out = ask(q)
    print(f"A: {out['answer']}\n")
    print(f"--- {out['n_steps']} messages, {len(out['trace'])} tool calls ---")
    for t in out["trace"]:
        print(f"  {t['tool']}({t['args']})")
