"""Shared structured logging for the RAG pipeline.

WHY THIS FILE EXISTS
--------------------
Both the app (`app.py`) and the CLI debugger (`debug.py`) need to record what
happened on each query. Centralizing it here means they log in the same format,
to the same file, with one source of truth — so you can debug real app usage and
scripted tests from a single `rag.log`.

WHAT IT RECORDS
---------------
For every query: the question, each retrieved chunk with its similarity score and
page, and the final answer. This is the single most useful artifact when asking
"why did it answer that?" — it shows whether the right context was retrieved.

USAGE
-----
    from rag_logging import log_retrieval
    scored = db.similarity_search_with_score(question, k=4)  # [(Document, score), ...]
    # ... build context, call the LLM, get `answer` ...
    log_retrieval(question, scored, answer)

Output goes to both the console and a rotating `rag.log` (1 MB x 3 backups, so it
never grows unbounded). Override the path with the RAG_LOG_FILE env var.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

_LOGGER_NAME = "rag"
_configured = False


def get_logger():
    """Return the singleton 'rag' logger, configuring handlers once."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if _configured:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Resolved at call time (not import time) so tests can redirect it via the
    # RAG_LOG_FILE env var before the first get_logger() call.
    log_file = os.getenv("RAG_LOG_FILE", "rag.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    _configured = True
    return logger


def log_retrieval(question, scored_docs, answer=None):
    """Log one query: the question, each retrieved chunk + score, and the answer.

    `scored_docs` is a list of (Document, score) tuples as returned by
    Chroma.similarity_search_with_score (lower score = closer for L2 distance).
    """
    logger = get_logger()
    # Flatten newlines so each logical entry stays on one line in the log.
    logger.info("QUERY: %s", question.replace("\n", " "))
    logger.info("RETRIEVED %d chunks:", len(scored_docs))
    for i, (doc, score) in enumerate(scored_docs, 1):
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:200].replace("\n", " ")
        logger.info("  [%d] score=%.4f page=%s | %s", i, score, page, preview)
    if answer is not None:
        logger.info("ANSWER: %s", answer.replace("\n", " "))
    logger.info("-" * 60)
