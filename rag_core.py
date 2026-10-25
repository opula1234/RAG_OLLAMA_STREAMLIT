"""Pure, framework-free RAG pipeline helpers.

WHY THIS FILE EXISTS
--------------------
The real pipeline logic (how PDFs are chunked, how retrieved chunks become
context, how the prompt is assembled) was previously inlined in `app.py` — which
can't be imported in tests because it's a Streamlit script that runs on import,
and in `debug.py`, which talks to a live Ollama. Extracting the logic here gives
us functions with NO Streamlit and NO Ollama dependencies, so they can be unit
tested in isolation. Both `app.py` and `debug.py` import from here, so there's a
single source of truth and no duplicated logic.

WHAT'S HERE
-----------
    split_documents  - chunk LangChain Documents for embedding
    build_context    - turn retrieved (Document, score) pairs into a prompt string
    build_prompt     - assemble the final LLM prompt from context + question

USAGE
-----
    from rag_core import split_documents, build_context, build_prompt
    chunks = split_documents(docs)
    context = build_context(scored_docs)        # [(Document, score), ...]
    prompt = build_prompt(context, "my question")
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Defaults kept here so app.py, debug.py, and the tests all agree on one value.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# The instruction that grounds the model in the retrieved context. Centralized so
# the app and the CLI debugger generate identical prompts (what you test is what
# you ship).
PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the question using ONLY the context below. "
    "If the answer is not in the context, say you don't know based on the document.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


def split_documents(docs, chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Split LangChain Documents into overlapping chunks for embedding.

    Smaller chunks give more precise retrieval; the overlap keeps facts that
    straddle a chunk boundary from being cut in half.

    Args:
        docs: list of langchain_core Document objects (e.g. from PyPDFLoader).
        chunk_size: max characters per chunk.
        chunk_overlap: characters shared between consecutive chunks.

    Returns:
        A list of chunked Document objects (metadata is preserved).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def build_context(scored_docs):
    """Join retrieved chunks into a single context string for the prompt.

    Args:
        scored_docs: list of (Document, score) tuples as returned by
            Chroma.similarity_search_with_score. The score is ignored here; it's
            kept in the pipeline for logging/display.

    Returns:
        The chunks' text joined by blank lines.
    """
    return "\n\n".join(doc.page_content for doc, _ in scored_docs)


def build_prompt(context, question):
    """Fill the grounding template with the retrieved context and the question.

    Args:
        context: the joined chunk text (see build_context).
        question: the user's natural-language question.

    Returns:
        The final prompt string to send to the LLM.
    """
    return PROMPT_TEMPLATE.format(context=context, question=question)
