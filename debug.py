"""Standalone, command-line diagnostics for the RAG pipeline.

WHY THIS FILE EXISTS
--------------------
RAG fails quietly — it returns confident but wrong answers. This script lets you
exercise each stage of the pipeline *independently of the Streamlit UI*, so you
can pinpoint exactly where things break instead of guessing. Each stage prints
what it actually sees (page/char counts, sample chunks, retrieved chunks with
similarity scores, the raw LLM answer).

THE "LAYERS" IT TESTS
---------------------
    Layer 0  Environment  - is Ollama running and are the models pulled?
    Layer 1  Ingestion    - does the PDF yield real text + chunks?
    Layer 2  Embedding    - does the embed model produce vectors?
    Layer 3  Retrieval    - do the right chunks come back for a question?
    Layer 4  Generation   - does the LLM answer well from those chunks?

The Layer 3 output is the key debugging signal: if the right chunks aren't
retrieved, it's a retrieval problem (tune chunking / top-k); if they ARE but the
answer is wrong, it's a generation/prompt problem.

USAGE
-----
    uv run debug.py                      # Layer 0 + 2: env + embedding checks
    uv run debug.py ingest mydoc.pdf     # Layer 1: extraction + chunking + store
    uv run debug.py query "your question" # Layer 3 + 4: retrieval + generation
"""
import os
import sys

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:8b")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")


def check_env():
    """Layer 0: verify Ollama is reachable and the required models are pulled.

    Hits Ollama's `/api/tags` endpoint directly (no heavy deps) and checks that
    both EMBED_MODEL and CHAT_MODEL are present, printing a `ollama pull ...` hint
    for any that are missing. Returns True only if Ollama responded, so callers
    can skip later checks when the environment is broken.
    """
    print("=== Layer 0: Environment ===")
    import urllib.request
    import json

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            tags = json.load(r)
        names = [m["name"] for m in tags.get("models", [])]
        print(f"[OK] Ollama is running. Models: {names}")
        for needed in (EMBED_MODEL, CHAT_MODEL):
            tag = needed if ":" in needed else needed + ":latest"
            ok = any(tag == n or n.startswith(needed) for n in names)
            print(f"  [{'OK' if ok else 'MISSING'}] {needed}"
                  + ("" if ok else f"  -> run: ollama pull {needed}"))
    except Exception as e:
        print(f"[FAIL] Cannot reach Ollama at :11434 ({e}). Is `ollama` running?")
        return False
    return True


def test_embedding():
    """Layer 2: confirm the embedding model actually produces a vector.

    Embeds a trivial string and prints the vector's dimension (e.g. 768 for
    nomic-embed-text). A failure here means the embed model is missing or Ollama
    is misbehaving — and nothing downstream (ingestion/retrieval) can work.
    """
    print("\n=== Layer 2: Embedding sanity ===")
    from langchain_ollama import OllamaEmbeddings

    vec = OllamaEmbeddings(model=EMBED_MODEL).embed_query("hello world")
    print(f"[OK] Embedding dim = {len(vec)}; first 3 values = {vec[:3]}")


def ingest(path):
    """Layer 1: load a PDF, report extraction quality, chunk it, and store it.

    This is the diagnostic twin of the app's sidebar ingestion. It prints page
    count, total extracted characters (near-zero => a scanned/image PDF that
    needs OCR), the number of chunks, and a sample chunk so you can eyeball that
    real text came out. Then it embeds and persists the chunks to ChromaDB so a
    subsequent `query` call has something to search.

    Args:
        path: filesystem path to a PDF file.
    """
    print(f"\n=== Layer 1: Ingestion of {path} ===")
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    from rag_core import split_documents

    docs = PyPDFLoader(path).load()
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"Pages loaded: {len(docs)}  |  total characters: {total_chars}")
    if total_chars < 50:
        print("[WARN] Almost no text extracted — likely a SCANNED/image PDF. "
              "You need OCR (unstructured[pdf] or pytesseract).")
    chunks = split_documents(docs)
    print(f"Chunks created: {len(chunks)}")
    if chunks:
        print(f"--- sample chunk[0] ---\n{chunks[0].page_content[:400]}\n---")
    Chroma.from_documents(chunks, OllamaEmbeddings(model=EMBED_MODEL),
                          persist_directory=PERSIST_DIR)
    print(f"[OK] Stored to {PERSIST_DIR}")


def query(q):
    """Layers 3+4: retrieve chunks for a question, then generate an answer.

    Prints the top-k retrieved chunks WITH their similarity scores (Layer 3) and
    the LLM's answer built from only those chunks (Layer 4), then logs the whole
    exchange to rag.log. Comparing the two tells you where a bad answer came from:
    wrong chunks retrieved = retrieval problem; right chunks but bad answer =
    generation/prompt problem. Requires an existing index (run `ingest` first).

    Args:
        q: the natural-language question to ask.
    """
    print(f"\n=== Layer 3+4: Retrieval + Generation for: {q!r} ===")
    from langchain_chroma import Chroma
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    from rag_core import build_context, build_prompt
    from rag_logging import log_retrieval

    if not os.path.exists(PERSIST_DIR):
        print("[FAIL] No index found. Run `uv run debug.py ingest <pdf>` first.")
        return
    db = Chroma(persist_directory=PERSIST_DIR,
                embedding_function=OllamaEmbeddings(model=EMBED_MODEL))

    scored = db.similarity_search_with_score(q, k=4)
    print("--- Layer 3: retrieved chunks (with scores) ---")
    for i, (doc, score) in enumerate(scored, 1):
        print(f"[{i}] score={score:.4f} page={doc.metadata.get('page')}  "
              f"{doc.page_content[:150]!r}")

    # Use the SAME context/prompt builders as the app so the CLI reproduces what
    # the app actually sends to the model.
    context = build_context(scored)
    print("\n--- Layer 4: LLM answer ---")
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.1)
    answer = llm.invoke(build_prompt(context, q)).content
    print(answer)

    log_retrieval(q, scored, answer)


if __name__ == "__main__":
    # Tiny CLI dispatcher: no args -> env + embedding checks; `ingest <pdf>` or
    # `query <text>` run the matching stage; anything else prints this usage.
    args = sys.argv[1:]
    if not args:
        if check_env():
            test_embedding()
    elif args[0] == "ingest":
        ingest(args[1])
    elif args[0] == "query":
        query(" ".join(args[1:]))
    else:
        print(__doc__)
