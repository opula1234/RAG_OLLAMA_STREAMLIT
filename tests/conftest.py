"""Shared pytest fixtures and helpers.

Keeps the unit tests fast and hermetic: no Streamlit, no running Ollama. The
`reset_rag_logger` fixture redirects the logger to a temp file and resets its
one-time-configured state so each test starts clean; `is_ollama_up` lets the
optional integration test skip itself when Ollama isn't running.
"""
import logging
import urllib.request

import pytest
from langchain_core.documents import Document


@pytest.fixture
def make_doc():
    """Factory for LangChain Documents, so tests don't repeat the boilerplate."""
    def _make(text, page=0):
        return Document(page_content=text, metadata={"page": page})
    return _make


@pytest.fixture
def reset_rag_logger(tmp_path, monkeypatch):
    """Point the 'rag' logger at a temp file and reset its singleton state.

    Yields the Path to the temp log file so a test can read it back and assert on
    the exact lines written.
    """
    import rag_logging

    log_file = tmp_path / "test_rag.log"
    monkeypatch.setenv("RAG_LOG_FILE", str(log_file))

    rag_logging._configured = False
    logger = logging.getLogger("rag")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    yield log_file

    # Tear down handlers so the next test (and real app) reconfigures cleanly.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    rag_logging._configured = False


def is_ollama_up():
    """Return True if Ollama answers on localhost:11434 (for integration skips)."""
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False
