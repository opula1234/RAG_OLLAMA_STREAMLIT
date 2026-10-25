# 📄 PDF RAG — Project Guide

A fully **open-source, fully local** Retrieval-Augmented Generation (RAG) app.
Upload a PDF, it gets chunked and embedded into a local vector store, and you
chat with its contents — no API keys, no cloud, no data leaving your machine.

---

## 1. What this project does

```
┌──────────────┐   PDF upload    ┌─────────────────────────────┐
│  Streamlit   │ ───────────────▶│  Ingestion pipeline          │
│  chat UI     │                 │  1. Extract text (pypdf)     │
│              │                 │  2. Split into chunks        │
│              │                 │  3. Embed (Ollama)           │
│              │                 │  4. Store vectors (Chroma)   │
└──────┬───────┘                 └─────────────────────────────┘
       │ user query
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Retrieval + Generation (RAG)                                │
│  1. Embed query (Ollama)                                     │
│  2. Similarity search in Chroma → top-k chunks               │
│  3. Stuff chunks into prompt + question                      │
│  4. LLM generates answer (Ollama) with sources + scores     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Tech stack

| Component      | Choice              | Notes                                   |
|----------------|---------------------|-----------------------------------------|
| LLM runtime    | **Ollama**          | Local, free, no API key                 |
| Chat model     | `llama3.1:8b`       | Good quality/size balance (16 GB RAM)   |
| Embeddings     | `nomic-embed-text`  | 768-dim, fast                           |
| Vector DB      | **ChromaDB**        | Local, in-process, zero config          |
| PDF parsing    | `pypdf`             | Text-based PDFs                         |
| Orchestration  | **LangChain**       | Glues the pipeline together             |
| UI             | **Streamlit**       | Chat interface                          |
| Package manager| **uv**              | Fast, reproducible via `uv.lock`        |

---

## 3. Project layout

```
RAG/
├── app.py              # The Streamlit RAG app (UI + ingestion + chat)
├── debug.py            # Standalone CLI to test the pipeline layer-by-layer
├── rag_logging.py      # Shared logger → writes rag.log
├── pyproject.toml      # Dependencies (uv)
├── uv.lock             # Pinned exact versions (commit this)
├── rag_core.py         # Pure, framework-free pipeline helpers (unit-tested)
├── README.md           # Quick start
├── PROJECT_GUIDE.md    # This file — full details, run & debug
├── pyproject.toml      # Dependencies + pytest config
├── .gitignore
├── tests/
│   ├── conftest.py         # Shared fixtures (temp logger, Ollama check)
│   ├── test_rag_core.py    # Chunking + context/prompt building
│   ├── test_rag_logging.py # Query-log format
│   └── test_integration.py # Optional: needs a running Ollama
└── .vscode/
    ├── launch.json     # F5 breakpoint-debug configs
    └── settings.json   # Points VS Code at the .venv
```

Generated at runtime (git-ignored): `.venv/`, `chroma_db/` (the vector store),
`rag.log*` (query logs), `.pytest_cache/`.

---

## 4. How to run

### Run with Docker Compose

Install Docker Desktop with Compose enabled, then from the project directory:

```powershell
docker compose up --build -d
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
```

Open <http://localhost:8501> and upload a PDF. The Compose configuration runs
the Streamlit app and Ollama in separate containers. It persists Chroma data
and logs in the `app-data` volume, and Ollama models in `ollama-data`.

```powershell
docker compose logs -f app
docker compose down       # stop containers and keep volumes
docker compose down -v    # also delete stored vectors, logs, and models
```

To use custom models, set `CHAT_MODEL` and/or `EMBED_MODEL` in the environment
before starting Compose, then pull those models into the Ollama container. Set
`OLLAMA_BASE_URL` only when running outside Compose; Compose configures it to
the Ollama service address automatically. Changing the embedding model requires
clearing the index and re-ingesting documents.

### Run directly with Ollama installed on the host

### Prerequisites

1. **Install Ollama** → https://ollama.com/download, then pull the models:
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```
   Verify both appear:
   ```bash
   ollama list
   ```
   Ollama must be **running** (it listens on `http://localhost:11434`).

2. **Install uv** (if not already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### Set up the environment

```bash
cd /Users/balajior/source/learning/RAG
uv sync          # creates .venv and installs everything from uv.lock
```

### Launch the app

```bash
uv run streamlit run app.py
```

A browser tab opens. In the sidebar:
1. **Upload a PDF** → click **Process PDF** (watch for "Indexed N chunks").
2. Ask questions in the chat box.
3. Expand **📚 Sources** under each answer to see which chunks (and similarity
   scores) were used.

Use **🗑️ Clear index** in the sidebar to wipe the vector store and start over.

### Configuration (optional)

Override models/paths via environment variables:

```bash
export CHAT_MODEL=qwen2.5:7b          # default: llama3.1:8b
export EMBED_MODEL=bge-m3             # default: nomic-embed-text
export PERSIST_DIR=./chroma_db        # default: ./chroma_db
export RAG_LOG_FILE=rag.log           # default: rag.log
```

> ⚠️ If you change `EMBED_MODEL`, **clear the index** and re-upload — embeddings
> from different models are not compatible (you'll get dimension mismatches).

Tuning knobs live at the top of `app.py`: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`.

---

## 5. How to debug

RAG fails *quietly* — it gives confident but wrong answers. Debug each stage in
isolation, bottom-up. There are three tools: the **CLI debugger** (`debug.py`),
the **query log** (`rag.log`), and **VS Code breakpoints**.

### 5a. CLI debugger — test without the UI

Isolates pipeline logic from UI issues. Run each layer independently:

```bash
uv run debug.py                        # Layer 0+2: Ollama up? models present? embeddings work?
uv run debug.py ingest yourfile.pdf    # Layer 1: pages/chars/chunks + a sample chunk
uv run debug.py query "your question"  # Layer 3+4: retrieved chunks + scores, then the answer
```

`debug.py query` is the key tool: it prints the **similarity scores** and the
**exact chunks** retrieved, so you instantly see whether a bad answer is a
*retrieval* problem (wrong chunks) or a *generation* problem (right chunks, bad
answer).

### 5b. Query log — watch real usage

Every query (from the app *and* from `debug.py`) is logged to `rag.log`:

```bash
tail -f rag.log
```

Example entry:
```
2026-10-25 01:06:40 [INFO] QUERY: What is the refund policy?
2026-10-25 01:06:40 [INFO] RETRIEVED 4 chunks:
2026-10-25 01:06:40 [INFO]   [1] score=0.2841 page=3 | Refunds are issued within 30 days...
2026-10-25 01:06:40 [INFO]   [2] score=0.5512 page=7 | Shipping costs are non-refundable...
2026-10-25 01:06:40 [INFO] ANSWER: Refunds are issued within 30 days of purchase...
```

**Reading the scores:** ChromaDB uses L2 distance by default — **lower = more
similar**. If the top score is high (e.g. > 1.0) or the relevant text isn't in
the list, retrieval is the culprit.

### 5c. VS Code breakpoints

1. Run `uv sync`, then reload VS Code (`Cmd+Shift+P` → *Developer: Reload
   Window*) so Pylance uses `.venv` (import warnings disappear).
2. Set a breakpoint (click the gutter left of a line number).
3. Open **Run & Debug** (`Cmd+Shift+D`), pick a config, press **F5**:
   - **Debug Streamlit app.py** — step through the live app
   - **Debug debug.py (env checks)** — step through Layer 0/2
   - **Debug debug.py ingest \<PDF\>** — step through extraction/chunking
   - **Debug debug.py query \<question\>** — step through retrieval/generation
4. At a breakpoint, use the **Debug Console** to inspect: `len(scored)`,
   `scored[0]`, `context`, etc.

> Two gotchas these configs already handle: Streamlit must launch as a **module**
> (not a script) or breakpoints never bind; and `justMyCode: false` lets you step
> into LangChain internals.

### 5d. Symptom → cause reference

| What you see | Likely layer | First thing to try |
|---|---|---|
| Import errors in IDE | Environment | select `.venv` interpreter; reload window |
| `Connection refused` on `:11434` | Environment | start Ollama |
| `model "..." not found` | Environment | `ollama pull <model>` |
| "Indexed 0 chunks" / near-zero chars | Ingestion | scanned/image PDF → needs OCR |
| Dimension mismatch error | Embedding/store | clear `chroma_db/`, re-ingest |
| Wrong answer, **sources look wrong** | Retrieval | lower `CHUNK_SIZE`, raise `TOP_K` |
| Wrong answer, **sources look right** | Generation | tighten prompt, keep temp low |
| Chat history glitches | UI/state | check `st.session_state.messages` |

---

## 6. Testing (pytest)

Tests live in `tests/` and run against the pure logic in `rag_core.py` and
`rag_logging.py` — so the default run needs **neither Streamlit nor Ollama** and
finishes in well under a second.

```bash
uv run pytest                 # unit tests (fast, hermetic) + skips integration if Ollama is down
uv run pytest -v              # verbose, one line per test
uv run pytest -m integration  # ONLY the Ollama-backed integration tests
uv run pytest -m "not integration"   # explicitly skip integration
```

What's covered:

| File | Tests |
|---|---|
| `test_rag_core.py` | Chunking (size limits, overlap, metadata preserved), context joining, prompt assembly |
| `test_rag_logging.py` | Log captures question/chunks/scores/answer; newline flattening; optional answer |
| `test_integration.py` | Real Ollama embeddings (auto-skipped when Ollama isn't running) |

Why the code is split this way: `app.py` is a Streamlit script (runs on import)
and `debug.py` talks to a live Ollama, so neither is unit-testable directly. The
testable logic was extracted into **`rag_core.py`**, which both import — one
source of truth that the tests pin down.

> The `integration` marker is registered in `pyproject.toml`. Integration tests
> skip themselves automatically if nothing answers on `localhost:11434`, so CI
> without Ollama stays green.

---

## 7. Swapping components

- **Different chat model:** `ollama pull qwen2.5:7b` then `export CHAT_MODEL=qwen2.5:7b`.
- **Qdrant** (open source, scalable): `uv add langchain-qdrant qdrant-client`.
- **Pinecone** (managed cloud — *not* open source): `uv add langchain-pinecone`
  and set `PINECONE_API_KEY`. Only the `Chroma(...)` lines in `app.py` change.

---

## 8. Common next steps

- **Multi-PDF support** — upload several, filter answers by document.
- **Conversation memory** — feed chat history so follow-up questions have context.
- **OCR** — handle scanned PDFs via `unstructured[pdf]` or `pytesseract`.
- **Reranking** — add a cross-encoder reranker to sharpen retrieval.
