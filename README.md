# 📄 Local PDF RAG (Ollama + ChromaDB + Streamlit)

A fully open-source, fully local Retrieval-Augmented Generation app. Upload a PDF,
it gets chunked and embedded into a local ChromaDB vector store, and you chat with
its contents — all running on your machine, no API keys, no data leaving your laptop.

## Stack

| Component    | Choice              |
|--------------|---------------------|
| LLM runtime  | Ollama              |
| Chat model   | `llama3.1:8b`       |
| Embeddings   | `nomic-embed-text`  |
| Vector DB    | ChromaDB (local)    |
| PDF parsing  | pypdf               |
| Orchestration| LangChain           |
| UI           | Streamlit           |

## Setup

### Option A: Docker Compose (recommended)

Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/)
with the Compose plugin enabled. From the project folder, build and start the
app and Ollama:

```powershell
docker compose up --build -d
```

Download the models into the persistent Ollama container volume (first run only):

```powershell
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
```

Open <http://localhost:8501>, upload a PDF, and click **Process PDF**. The app
and Ollama run as separate containers; Docker Compose connects them on a private
network. Chroma data, query logs, and Ollama models persist in named Docker
volumes when containers are stopped or rebuilt.

Useful commands:

```powershell
docker compose logs -f app          # follow application logs
docker compose exec ollama ollama list
docker compose down                 # stop containers; keep stored data
docker compose down -v              # remove containers AND stored data
```

The app is published only on `127.0.0.1:8501`. The Ollama API is not published
to the host. To use different models, set `CHAT_MODEL` and/or `EMBED_MODEL`
before starting Compose, and pull those models into the Ollama container. If
you change the embedding model, clear the app's index and re-process documents.
The default Compose setup does not configure GPU passthrough; Ollama may run
more slowly on CPU.

### Option B: Run directly with Ollama installed on the host

### 1. Install & start Ollama
Download from https://ollama.com/download, then pull the models:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Make sure Ollama is running (`ollama list` should show both models).

### 2. Python environment (uv)

This project uses [uv](https://docs.astral.sh/uv/). Install it once
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), then:

```bash
uv sync
```

That creates `.venv` and installs everything from `pyproject.toml` / `uv.lock`.

### 3. Run

```bash
uv run streamlit run app.py
```

`uv run` auto-activates the environment, so you don't need to `source .venv/...`.

Open the browser tab, upload a PDF in the sidebar, click **Process PDF**, then ask away.

## Configuration

Override models via environment variables (defaults in parentheses):

```bash
export CHAT_MODEL=qwen2.5:7b          # (llama3.1:8b)
export EMBED_MODEL=bge-m3             # (nomic-embed-text)
export PERSIST_DIR=./chroma_db        # where vectors are stored
export OLLAMA_BASE_URL=http://localhost:11434
```

> ⚠️ If you change `EMBED_MODEL`, clear the index (🗑️ button) and re-upload — old
> and new embeddings are not compatible.

## Tuning tips

- **`CHUNK_SIZE` / `CHUNK_OVERLAP`** (in `app.py`): smaller chunks = more precise
  retrieval; larger = more context per chunk.
- **`TOP_K`**: how many chunks are fed to the LLM. Raise for broad questions.
- **`temperature`**: kept at `0.1` for factual answers.

## Testing

```bash
uv run pytest                 # fast unit tests (no Ollama needed)
uv run pytest -m integration  # Ollama-backed tests (auto-skipped if it's down)
```

See `PROJECT_GUIDE.md` §6 for details.

## Swapping in a different vector DB

- **Qdrant** (open source, scalable): `uv add langchain-qdrant qdrant-client`
- **Pinecone** (managed cloud, needs API key — not open source):
  `uv add langchain-pinecone` and set `PINECONE_API_KEY`.

Only the `Chroma.from_documents(...)` / `Chroma(...)` lines in `app.py` change.
