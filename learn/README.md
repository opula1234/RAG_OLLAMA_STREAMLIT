# 🧠 Learn RAG from scratch — plain Python, no frameworks

These scripts teach **what RAG actually is** by building it piece by piece in
plain Python. No Streamlit, no LangChain, no vector database — the similarity
search is even computed by hand so nothing is hidden. Once these click, the
"real" app (`../app.py`) is just the same ideas with faster/fancier tools.

## What is RAG, in one paragraph

A large language model (LLM) only knows what it was trained on, and it can make
things up. **RAG (Retrieval-Augmented Generation)** fixes this: before asking the
model, you *retrieve* the most relevant pieces of your own documents and paste
them into the prompt. The model then answers from YOUR text instead of guessing.
It's three steps — **embed → retrieve → generate**.

## The three ideas

| Idea | What it means | Lesson |
|------|---------------|--------|
| **Embedding** | Turn text into a list of numbers (a vector) where similar *meaning* → similar vector | `01_embeddings.py` |
| **Retrieval** | Embed a question, compare it to every chunk's vector, keep the closest ones | `02_retrieval.py` |
| **Generation** | Put the retrieved chunks + question in a prompt, let the LLM answer from them | `03_rag.py` |
| **Agentic RAG** | Add a loop/branch: if retrieval is weak, rewrite the question and retry (LangGraph) | `04_agentic_rag.py` |

## Run them in order

Make sure Ollama is running and the models are pulled:

```bash
ollama pull nomic-embed-text    # the embedding model
ollama pull llama3.1:8b         # the chat model (only needed for lesson 3)
```

Then:

```bash
uv run learn/01_embeddings.py   # see text become vectors + measure similarity
uv run learn/02_retrieval.py    # find the most relevant chunks for a question
uv run learn/03_rag.py          # the full pipeline: retrieve + generate an answer
uv run learn/04_agentic_rag.py  # add a loop: grade retrieval, rewrite & retry (LangGraph)
```

Each script prints every intermediate step so you can watch the data flow.

## How this maps to the "real" app

What you build here by hand, the production app does with libraries:

| In these lessons (by hand) | In `../app.py` (with tools) |
|----------------------------|------------------------------|
| `KNOWLEDGE_BASE` list of strings | text extracted from an uploaded **PDF** (pypdf) |
| `embed()` via raw `ollama` client | `OllamaEmbeddings` (LangChain) |
| `cosine_similarity()` + a Python loop | **ChromaDB** vector database (fast search) |
| `build_prompt()` string | LangChain `ChatPromptTemplate` |
| `generate()` via raw `ollama` client | `ChatOllama` (streamed to the UI) |
| `print()` statements | **Streamlit** chat interface |

Same pipeline — the libraries just make it faster and prettier.

## Suggested next steps (once the 3 lessons make sense)

1. **Chunk a real document** — replace `KNOWLEDGE_BASE` with text split from a
   `.txt` or PDF file, and see retrieval work on real content.
2. **Swap the hand-written search for ChromaDB** — same results, but scales to
   thousands of chunks.
3. **Add the UI** — wrap it in Streamlit (that's what `../app.py` already does).
