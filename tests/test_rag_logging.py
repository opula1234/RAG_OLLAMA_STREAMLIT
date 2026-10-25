"""Unit tests for the structured query logger (rag_logging).

Verifies the log file captures the question, every retrieved chunk with its
score and page, and the answer — the artifact you rely on when debugging
"why did it answer that?".
"""
from rag_logging import log_retrieval


def test_log_retrieval_writes_all_sections(reset_rag_logger, make_doc):
    scored = [(make_doc("refund within 30 days", page=3), 0.2841)]
    log_retrieval("What is the refund policy?", scored, "Refunds take 30 days.")

    content = reset_rag_logger.read_text()
    assert "QUERY: What is the refund policy?" in content
    assert "RETRIEVED 1 chunks" in content
    assert "score=0.2841" in content
    assert "page=3" in content
    assert "ANSWER: Refunds take 30 days." in content


def test_log_retrieval_logs_each_chunk(reset_rag_logger, make_doc):
    scored = [
        (make_doc("chunk one", page=1), 0.10),
        (make_doc("chunk two", page=2), 0.20),
        (make_doc("chunk three", page=3), 0.30),
    ]
    log_retrieval("q", scored, "a")

    content = reset_rag_logger.read_text()
    assert "RETRIEVED 3 chunks" in content
    assert "chunk one" in content and "chunk two" in content and "chunk three" in content


def test_answer_is_optional(reset_rag_logger, make_doc):
    log_retrieval("q", [(make_doc("x"), 0.5)])  # no answer passed
    content = reset_rag_logger.read_text()
    assert "QUERY: q" in content
    assert "ANSWER:" not in content


def test_newlines_in_content_are_flattened(reset_rag_logger, make_doc):
    scored = [(make_doc("line1\nline2", page=1), 0.5)]
    log_retrieval("multi\nline question", scored, "multi\nline answer")

    # Each logical log entry should stay on its own line (newlines flattened).
    content = reset_rag_logger.read_text()
    assert "line1 line2" in content
    assert "multi line question" in content
    assert "multi line answer" in content
