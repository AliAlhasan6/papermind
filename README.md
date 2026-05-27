
# PaperMind

> A local research assistant that answers questions about your papers — and cites every claim.

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Inference](https://img.shields.io/badge/inference-100%25%20local-0a7)

PaperMind ingests a folder of academic PDFs and lets you ask questions about them in plain language. Every answer is grounded in the documents you loaded and traced back to the exact source chunk it came from. It runs entirely on local hardware — no cloud APIs, no keys, no data leaving the machine.

---

## What makes it different

Most "chat with your PDFs" tools do one thing: semantic search. PaperMind does two, and lets an agent decide which to use.

- **`vector_search`** — semantic similarity. Answers *"what does the paper say about X?"*
- **`graph_neighbors`** — knowledge-graph traversal. Answers *"what is related to X?"*
- **`cite_source`** — fetches verbatim chunk text, so the agent can verify before answering.

A LangGraph ReAct agent reads each question, picks the appropriate tool (or both), and composes an answer where every factual sentence carries a citation. Ask a "what does it say" question and it reaches for vector search; ask a "what relates to what" question and it walks the knowledge graph. The routing is the point — it is a small demonstration of an agent reasoning about *how* to retrieve, not just retrieving.

## Screenshots

![PaperMind answering a question with citations and an agent trace](docs/screenshots/hero-dark.png)

*A cited answer, the Sources panel, the Agent trace showing which tools were used, and a clickable knowledge-graph viewer.*

![PaperMind light theme](docs/screenshots/hero-light.png)

*Light theme. PaperMind follows the OS preference and has a manual toggle.*

## Architecture

PaperMind has two phases: an **offline ingestion pipeline** that runs once per corpus, and an **online query pipeline** that runs per question. They share the vector store and the knowledge graph.



### Ingestion (run once per corpus)
Each PDF is parsed and split into overlapping chunks. The chunks fork down two paths: one is embedded into a Chroma vector store; the other is processed by a local LLM that extracts entities and relations into a NetworkX knowledge graph. The graph is then resolved (duplicate nodes merged) and cleaned (formula and OCR-noise nodes dropped). The extraction job is resumable — it checkpoints progress and can be interrupted and restarted without losing work.

```mermaid
flowchart TB
    %% ──────────── INGESTION ────────────
    subgraph INGEST["Ingestion (offline, once per corpus)"]
        direction TB
        PDF["PDFs<br/><i>binary</i>"]
        TXT["text<br/><i>utf-8 str</i>"]
        CHK["chunks<br/><i>~800 chars, 100 overlap</i>"]
        EMB["MiniLM embedder<br/><i>multilingual, CPU</i>"]
        LLM_EXT["Qwen 2.5 7B<br/><i>entity + relation prompt</i>"]
        VEC["vectors<br/><i>float32[384]</i>"]
        TRIPLES["JSON triples<br/><i>(head, relation, tail)</i>"]

        PDF -->|pypdf| TXT
        TXT -->|split| CHK
        CHK --> EMB
        CHK --> LLM_EXT
        EMB --> VEC
        LLM_EXT --> TRIPLES
    end

    %% ──────────── STORAGE ────────────
    subgraph STORE["Storage (shared)"]
        direction LR
        CHROMA[("Chroma<br/><i>persistent vector store</i>")]
        KG[("NetworkX<br/><i>kg.pkl</i>")]
    end

    VEC --> CHROMA
    TRIPLES -->|resolve + clean| KG

    %% ──────────── QUERY ────────────
    subgraph QUERY["Query (online, per question)"]
        direction TB
        Q["question<br/><i>utf-8 str</i>"]
        AGENT{"LangGraph agent<br/><i>ReAct loop</i>"}
        VS["vector_search<br/><i>top-k chunks</i>"]
        GN["graph_neighbors<br/><i>entity + relations</i>"]
        CS["cite_source<br/><i>verbatim chunk</i>"]
        ANS["answer<br/><i>str + [chunk_ids]</i>"]

        Q --> AGENT
        AGENT -->|semantic Q| VS
        AGENT -->|relational Q| GN
        AGENT -->|verify| CS
        VS -->|loop| AGENT
        GN -->|loop| AGENT
        CS -->|loop| AGENT
        AGENT --> ANS
    end

    CHROMA -.->|reads| VS
    CHROMA -.->|reads| CS
    KG -.->|reads| GN

    %% ──────────── UI ────────────
    UI["Browser UI<br/><i>FastAPI + vanilla HTML/CSS/JS</i>"]
    ANS --> UI
    UI -->|POST /chat| Q

    %% ──────────── ATLAS THEME ────────────
    classDef ingest fill:#f4f6ee,stroke:#134a48,stroke-width:1px,color:#141815
    classDef store fill:#e3dca4,stroke:#8a6815,stroke-width:1.5px,color:#141815
    classDef query fill:#c4d8d5,stroke:#134a48,stroke-width:1px,color:#0a312f
    classDef agent fill:#134a48,stroke:#134a48,stroke-width:1.5px,color:#f4f6ee
    classDef ui fill:#ebeee3,stroke:#7a8079,stroke-width:1px,color:#141815

    class PDF,TXT,CHK,EMB,LLM_EXT,VEC,TRIPLES ingest
    class CHROMA,KG store
    class Q,VS,GN,CS,ANS query
    class AGENT agent
    class UI ui
```

### Query (run per question)
The agent receives the question, selects and calls tools, optionally loops to retrieve more, and returns an answer grounded in chunk IDs. A FastAPI backend exposes this over HTTP; a vanilla HTML/CSS/JS frontend renders the chat, the citations, the agent trace, and a clickable force-directed view of the knowledge graph.


## Quickstart

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com).

```bash
# 1. Clone and install
git clone https://github.com/AliAlhasan6/papermind.git
cd papermind
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Pull the local model
ollama pull qwen2.5:7b-instruct-q4_K_M

# 3. Configure (defaults are sensible — no editing needed)
cp .env.example .env

# 4. Add PDFs and ingest them
cp /path/to/your/papers/*.pdf data/papers/
python -m papermind.ingest data/papers/

# 5. (Optional) Build the knowledge graph — enables graph_neighbors
python -m papermind.extract_all
python -m papermind.resolve
python -m papermind.clean_kg

# 6. Run
uvicorn papermind.api:app --port 8000
# open http://localhost:8000
```

Vector search works immediately after step 4. The knowledge-graph tool requires step 5, which calls the local LLM once per chunk — on modest hardware this is an overnight job for a large corpus, and it is resumable. PDFs can also be added and removed through the web UI.

## Design decisions

**Why fully local.** No API keys, no quotas, no data leaving the machine. The entire system — LLM, embeddings, vector store, knowledge graph — runs on commodity hardware (developed on a laptop with a 4 GB GPU). This makes it reproducible and suitable for privacy-sensitive corpora.

**Why hybrid retrieval.** Vector search is strong for semantic questions but weak for questions about *relationships* between concepts. Adding a knowledge-graph layer gives the agent a second mode of retrieval and a decision to make. The decision is the interesting part.

**Why a multilingual embedder.** The embedder was chosen so that English queries can retrieve relevant passages from a non-English corpus — cross-lingual retrieval from a single vector space.

**Why citations are mandatory.** The agent is instructed that an answer without chunk-ID citations is invalid. This is the enforcement mechanism for grounding: every claim must trace to a real chunk, and the UI lets you click any citation to read the verbatim source. The knowledge-graph viewer is likewise clickable — every node shows its neighbors, so the graph the agent traverses is inspectable.

## Limitations

PaperMind is an honest small system, not a polished product. Known limitations:

- **Knowledge-graph quality.** Entities and relations are extracted by a 7B local model. Roughly two-thirds to three-quarters of chunks yield usable extractions; relation types are approximate. The graph is best treated as a navigation aid, not a formal ontology.
- **Graph fragmentation.** Independent per-chunk extraction produces a fragmented graph. Entity resolution is currently string-level only. Semantic entity merging is future work.
- **Inference speed.** Local inference on a low-end GPU is slow — a single answer can take a few minutes. This is the deliberate cost of running fully offline.
- **Multilingual answers.** For a non-English corpus, answers are composed in English but source terms and entity names are preserved in their original language, since they are more precise and easier to verify against the source.

## Future work

- Semantic entity resolution (embed entity names, merge near-duplicates)
- Reranking after vector retrieval
- Word-document (`.docx`) ingestion
- A cloud-LLM configuration for a faster hosted demo
- An evaluation harness measuring retrieval recall and citation accuracy

## License

MIT — see [LICENSE](LICENSE).

Built by Alhasan Ali.
