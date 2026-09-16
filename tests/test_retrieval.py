import pytest

from rag_insight.models import Candidate, Chunk
from rag_insight.retrieval import bm25_search, dense_search, fuse, select_context


def chunk(key, text):
    return Chunk("doc", "v1", "test.md", text, chunk_id=key)


def test_rrf_merges_duplicate_ids_and_ignores_score_scales():
    a, b = chunk("a", "Redis"), chunk("b", "PostgreSQL")
    result = fuse([[Candidate(a, 0.2), Candidate(b, 0.1)], [Candidate(b, 9000)]])
    assert len(result) == 2
    assert result[0].chunk.chunk_id == "b"


def test_bm25_finds_identifier_and_excludes_zero_matches():
    rows = [(chunk("a", "Send X-API-Key header"), []), (chunk("b", "Redis caching"), [])]
    assert [c.chunk.chunk_id for c in bm25_search("X-API-Key", rows, 5)] == ["a"]


def test_dense_search_rejects_vector_dimension_mismatch():
    rows = [(chunk("a", "Redis"), [1.0, 0.0])]
    with pytest.raises(ValueError, match="dimensions do not match"):
        dense_search([1.0, 0.0, 0.0], rows, 5)


def test_context_prefers_a_specific_multi_term_match():
    table = chunk("table", "Tables and Data Display\n001 Jane Smith")
    generic = chunk("generic", "This document includes tables, images, and hyperlinks")
    settings = type("Settings", (), {"mmr": False, "context_k": 5, "context_tokens": 100,
                                     "mmr_lambda": 0.7})()
    tokenizer = type("Tokenizer", (), {"encode": lambda _, text, add_special_tokens=False: text.split()})()
    selected = select_context([Candidate(table, 1), Candidate(generic, 0.9)],
                              [(table, [1]), (generic, [1])], settings, tokenizer,
                              "what values are present inside the tables and displays?")
    assert [candidate.chunk.chunk_id for candidate in selected] == ["table"]


def test_aggregate_context_keeps_same_page_continuation():
    first = Chunk("doc", "v1", "versions.pdf", "PDF version and year", page=11, chunk_id="first")
    continuation = Chunk("doc", "v1", "versions.pdf", "PDF 2.0 2017", page=11, chunk_id="continuation")
    other = Chunk("doc", "v1", "versions.pdf", "Unrelated page", page=12, chunk_id="other")
    settings = type("Settings", (), {"mmr": False, "context_k": 2, "context_tokens": 100,
                                     "mmr_lambda": 0.7})()
    tokenizer = type("Tokenizer", (), {"encode": lambda _, text, add_special_tokens=False: text.split()})()
    selected = select_context([Candidate(first, 1), Candidate(other, 0.9), Candidate(continuation, 0.1)],
                              [(first, [1]), (continuation, [1]), (other, [1])], settings, tokenizer,
                              "list all PDF versions and years")
    assert [candidate.chunk.chunk_id for candidate in selected] == ["first", "continuation"]
