"""Fully-local PDF RAG (Retrieval-Augmented Generation) web app.

WHY THIS FILE EXISTS
--------------------
This is the main user-facing application. It lets a user upload a PDF and then
ask questions about it in a chat interface, with answers grounded in the PDF's
actual content (not the model's general knowledge). Everything runs locally:
no API keys, no cloud, no data leaving the machine.

HOW IT WORKS (two phases)
-------------------------
1. Ingestion (sidebar, runs once per PDF):
     PDF -> extract text (pypdf) -> split into overlapping chunks
         -> embed each chunk (Ollama) -> store vectors in ChromaDB.
2. Chat (main panel, runs per question):
     question -> embed + similarity search in ChromaDB (top-k chunks)
              -> stuff chunks into a prompt -> LLM generates an answer (Ollama)
              -> show answer + the source chunks/scores, and log the query.

NOTE ON STRUCTURE
-----------------
Streamlit re-runs this whole script top-to-bottom on every UI interaction. That
is why expensive objects (the embedding + chat clients) are wrapped in
`@st.cache_resource` so they are created once and reused across re-runs.

USAGE
-----
    uv run streamlit run app.py
Then upload a PDF in the sidebar, click "Process PDF", and ask questions.
"""
import os
import tempfile

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from rag_core import PROMPT_TEMPLATE, build_context, split_documents
from rag_logging import log_retrieval

# --- Config -----------------------------------------------------------------
# Models/paths are read from env vars (with sane defaults) so you can swap them
# without editing code, e.g. `export CHAT_MODEL=qwen2.5:7b`. The chunking/top-k
# knobs below are the main levers for tuning retrieval quality.
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")  # must match at query time
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:8b")         # answer generator
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")       # where vectors live on disk
CHUNK_SIZE = 1000       # characters per chunk; smaller = more precise retrieval
CHUNK_OVERLAP = 150     # overlap so facts spanning a boundary aren't split apart
TOP_K = 4               # how many chunks to retrieve and feed to the LLM

st.set_page_config(page_title="PDF RAG (Ollama)", page_icon="📄")
st.title("📄 Chat with your PDF (fully local)")
st.caption(f"LLM: `{CHAT_MODEL}`  ·  Embeddings: `{EMBED_MODEL}`  ·  Vector DB: ChromaDB")


@st.cache_resource
def get_embeddings():
    """Return a cached Ollama embedding client.

    Used for BOTH ingestion (embedding PDF chunks) and querying (embedding the
    user's question) — the same model must be used for both so the vectors are
    comparable. Cached with `@st.cache_resource` so the client is built once and
    survives Streamlit's per-interaction re-runs.
    """
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


@st.cache_resource
def get_llm():
    """Return a cached Ollama chat client used to generate answers.

    `temperature=0.1` keeps answers factual/deterministic, which is what we want
    for RAG (summarize the retrieved context, don't get creative). Cached so the
    connection is reused across re-runs.
    """
    return ChatOllama(
        model=CHAT_MODEL, temperature=0.1, base_url=OLLAMA_BASE_URL
    )


embeddings = get_embeddings()

# --- Sidebar: ingestion -----------------------------------------------------
# Turns an uploaded PDF into searchable vectors. PyPDFLoader needs a real file
# path, so the upload is written to a temp file (and deleted afterwards). The
# text is chunked, embedded, and persisted into ChromaDB on disk.
with st.sidebar:
    st.header("Upload & index")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded and st.button("Process PDF", type="primary"):
        with st.spinner("Extracting, chunking, embedding..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                path = tmp.name
            try:
                docs = PyPDFLoader(path).load()
                chunks = split_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
                Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
            finally:
                os.unlink(path)
        st.success(f"Indexed {len(chunks)} chunks from '{uploaded.name}'.")

    if os.path.exists(PERSIST_DIR) and st.button("🗑️ Clear index"):
        import shutil

        shutil.rmtree(PERSIST_DIR)
        st.session_state.pop("messages", None)
        st.rerun()

# --- Retrieval + chat -------------------------------------------------------
# From here on we need an existing index. If none exists yet, prompt the user to
# ingest a PDF first and stop the script early.
if not os.path.exists(PERSIST_DIR):
    st.info("👈 Upload a PDF and click **Process PDF** to begin.")
    st.stop()

db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
llm = get_llm()

# Prompt template is defined once in rag_core (shared with debug.py + tests).
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

if question := st.chat_input("Ask something about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    # similarity_search_with_score returns (Document, score) so we can log/show
    # the retrieval quality, not just the chunks.
    scored = db.similarity_search_with_score(question, k=TOP_K)
    context = build_context(scored)
    chain = prompt | llm

    with st.chat_message("assistant"):
        answer = st.write_stream(
            chunk.content for chunk in chain.stream(
                {"context": context, "question": question}
            )
        )
        with st.expander("📚 Sources"):
            for i, (doc, score) in enumerate(scored, 1):
                page = doc.metadata.get("page", "?")
                st.caption(f"Source {i} — page {page} · score {score:.4f}")
                st.text(doc.page_content[:300] + "...")

    log_retrieval(question, scored, answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
