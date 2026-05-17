# PaperMind

> Hybrid RAG + knowledge-graph research agent. **100% local inference** via Ollama. Cites every claim.

🚧 Under construction — Week 1 build in progress.

## Why local

- **No API keys, no quotas, no data leaves your machine.** Reproducible on any laptop with a 4 GB GPU.
- Demonstrates the full stack runs on commodity hardware — relevant for privacy-sensitive RAG (legal, medical, research).
- Llama-3.1 8B / Qwen 2.5 7B (Q4 quantized) handles ReAct + JSON-structured KG extraction at ~15-25 tok/s on a GTX 1050 Ti.

## Architecture (planned)

- **Inference:** Ollama (Qwen 2.5 7B Q4) on `localhost:11434`
- **Ingestion:** PDFs → chunks (Chroma) + entities/relations (NetworkX, LLM-extracted)
- **Agent:** LangGraph ReAct with three tools — `vector_search`, `graph_neighbors`, `cite_source`
- **UI:** Gradio (local) + Hugging Face Spaces deploy (CPU fallback config)

## Stack

Free / open-source throughout: Ollama, Qwen 2.5, Chroma, sentence-transformers, NetworkX, LangGraph, Gradio.

## License

MIT
