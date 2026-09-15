from rag_insight.models import Candidate, Chunk
from rag_insight.retrieval import bm25_search, fuse


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
