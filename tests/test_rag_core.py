"""Unit tests for the pure pipeline helpers in rag_core.

These need neither Streamlit nor a running Ollama — they exercise chunking and
the context/prompt string-building that sit at the heart of the RAG pipeline.
"""
from rag_core import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    PROMPT_TEMPLATE,
    build_context,
    build_prompt,
    split_documents,
)


class TestSplitDocuments:
    def test_long_document_splits_into_multiple_chunks(self, make_doc):
        doc = make_doc("word " * 1000)  # ~5000 chars, well over one chunk
        chunks = split_documents([doc], chunk_size=1000, chunk_overlap=150)
        assert len(chunks) > 1

    def test_chunks_respect_size_limit(self, make_doc):
        doc = make_doc("abcdefghij " * 500)
        chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)
        assert all(len(c.page_content) <= 200 for c in chunks)

    def test_short_document_stays_single_chunk(self, make_doc):
        chunks = split_documents([make_doc("just a short sentence.")])
        assert len(chunks) == 1

    def test_metadata_is_preserved(self, make_doc):
        chunks = split_documents([make_doc("word " * 1000, page=7)])
        assert all(c.metadata["page"] == 7 for c in chunks)

    def test_defaults_are_applied(self, make_doc):
        # Smoke test that the module defaults are usable without explicit args.
        assert DEFAULT_CHUNK_SIZE > DEFAULT_CHUNK_OVERLAP
        chunks = split_documents([make_doc("hello world")])
        assert len(chunks) == 1


class TestBuildContext:
    def test_joins_chunks_with_blank_lines(self, make_doc):
        scored = [(make_doc("first"), 0.1), (make_doc("second"), 0.2)]
        assert build_context(scored) == "first\n\nsecond"

    def test_ignores_scores(self, make_doc):
        scored = [(make_doc("only"), 99.9)]
        assert build_context(scored) == "only"

    def test_empty_input_gives_empty_string(self):
        assert build_context([]) == ""


class TestBuildPrompt:
    def test_includes_context_and_question(self):
        prompt = build_prompt("the context text", "the question?")
        assert "the context text" in prompt
        assert "the question?" in prompt

    def test_matches_shared_template(self):
        # The app and debugger must produce the exact same prompt string.
        expected = PROMPT_TEMPLATE.format(context="C", question="Q")
        assert build_prompt("C", "Q") == expected
