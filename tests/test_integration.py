"""Optional integration tests that require a running Ollama.

These are marked `integration` and skipped automatically when Ollama isn't
reachable, so the default `pytest` run stays fast and hermetic. Run them
explicitly with:  uv run pytest -m integration
"""
import pytest

from conftest import is_ollama_up

pytestmark = pytest.mark.integration

EMBED_MODEL = "nomic-embed-text"


@pytest.mark.skipif(not is_ollama_up(), reason="Ollama not running on :11434")
def test_embedding_produces_vector():
    from langchain_ollama import OllamaEmbeddings

    vec = OllamaEmbeddings(model=EMBED_MODEL).embed_query("hello world")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(x, float) for x in vec[:5])


@pytest.mark.skipif(not is_ollama_up(), reason="Ollama not running on :11434")
def test_embeddings_are_consistent_length():
    from langchain_ollama import OllamaEmbeddings

    emb = OllamaEmbeddings(model=EMBED_MODEL)
    a = emb.embed_query("first sentence")
    b = emb.embed_query("a completely different second sentence")
    assert len(a) == len(b)  # same model -> same dimensionality
