import sqlite3
from threading import Thread

import pytest

from rag_insight.models import Chunk
from rag_insight.storage import ChromaStore, Store


def make_chunk(chunk_id, document_id="document", text="text"):
    return Chunk(document_id, "v1", f"{document_id}.md", text, line_start=1, line_end=1, chunk_id=chunk_id)


def test_insert_read_and_close_reopen_round_trip(tmp_path):
    path = tmp_path / "index.sqlite"
    with Store(path) as store:
        store.replace_document("document", [make_chunk("one")], [[0.1, 0.2]], "signature")
        assert store.signature() == "signature"
        assert store.all()[0][1] == [0.1, 0.2]
    with Store(path) as store:
        rows = store.all()
        assert rows[0][0].chunk_id == "one"
        assert store.signature() == "signature"


def test_replacement_only_removes_target_document(tmp_path):
    with Store(tmp_path / "index.sqlite") as store:
        store.replace_document("one", [make_chunk("old", "one")], [[1]], "signature")
        store.replace_document("two", [make_chunk("other", "two")], [[2]], "signature")
        store.replace_document("one", [make_chunk("new", "one")], [[3]], "signature")
        assert {chunk.chunk_id for chunk, _ in store.all()} == {"new", "other"}


@pytest.mark.parametrize("chunks,vectors", [([], []), ([make_chunk("one")], [])])
def test_invalid_batches_do_not_mutate_existing_data(tmp_path, chunks, vectors):
    with Store(tmp_path / "index.sqlite") as store:
        store.replace_document("document", [make_chunk("old")], [[1]], "signature")
        with pytest.raises(ValueError, match="matching embeddings"):
            store.replace_document("document", chunks, vectors, "signature")
        assert [chunk.chunk_id for chunk, _ in store.all()] == ["old"]


def test_signature_mismatch_is_rejected_before_mutation(tmp_path):
    with Store(tmp_path / "index.sqlite") as store:
        store.replace_document("document", [make_chunk("old")], [[1]], "first")
        with pytest.raises(ValueError, match="configuration differs"):
            store.replace_document("document", [make_chunk("new")], [[2]], "second")
        assert store.signature() == "first"
        assert [chunk.chunk_id for chunk, _ in store.all()] == ["old"]


def test_failed_insert_rolls_back_deletion_and_config_change(tmp_path):
    with Store(tmp_path / "index.sqlite") as store:
        store.replace_document("document", [make_chunk("old")], [[1]], "signature")
        duplicate = [make_chunk("duplicate"), make_chunk("duplicate", text="other")]
        with pytest.raises(sqlite3.IntegrityError):
            store.replace_document("document", duplicate, [[2], [3]], "signature")
        assert [chunk.chunk_id for chunk, _ in store.all()] == ["old"]


def test_store_can_be_read_from_another_thread(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    store.replace_document("document", [make_chunk("one")], [[1, 0]], "signature")
    result = []
    worker = Thread(target=lambda: result.append(store.signature()))
    worker.start()
    worker.join()
    store.close()
    assert result == ["signature"]


def test_chroma_store_persists_replaces_and_searches(tmp_path):
    path = tmp_path / "chroma"
    chunks = [
        make_chunk("redis", "one", "Redis is used for caching"),
        make_chunk("api", "two", "Send the X-API-Key header"),
    ]
    for chunk in chunks:
        chunk.content_hash = f"hash-{chunk.chunk_id}"
        chunk.embedding_model = "test-model"
        chunk.embedding_dimension = 2
    with ChromaStore(path, "test-rag") as store:
        store.replace_document("one", [chunks[0]], [[1.0, 0.0]], "signature")
        store.replace_document("two", [chunks[1]], [[0.0, 1.0]], "signature")
        assert [candidate.chunk.chunk_id for candidate in store.dense_search([1.0, 0.0], 1)] == ["redis"]
        assert store.cached_vectors(["hash-api"], "test-model") == {"hash-api": [0.0, 1.0]}
        replacement = make_chunk("redis-new", "one", "Redis replacement")
        replacement.content_hash = "hash-redis-new"
        replacement.embedding_model = "test-model"
        replacement.embedding_dimension = 2
        store.replace_document("one", [replacement], [[1.0, 0.0]], "signature")
        assert {chunk.chunk_id for chunk, _ in store.all()} == {"redis-new", "api"}
    with ChromaStore(path, "test-rag") as reopened:
        assert reopened.signature() == "signature"
        assert {chunk.chunk_id for chunk, _ in reopened.all()} == {"redis-new", "api"}


def test_chroma_store_rejects_incompatible_signature_before_mutation(tmp_path):
    with ChromaStore(tmp_path / "chroma", "test-rag") as store:
        chunk = make_chunk("old")
        chunk.content_hash, chunk.embedding_model, chunk.embedding_dimension = "hash", "test", 1
        store.replace_document("document", [chunk], [[1.0]], "first")
        with pytest.raises(ValueError, match="configuration differs"):
            store.replace_document("document", [make_chunk("new")], [[1.0]], "second")
        assert [chunk.chunk_id for chunk, _ in store.all()] == ["old"]


@pytest.mark.parametrize("factory", [
    lambda path: Store(path / "index.sqlite"),
    lambda path: ChromaStore(path / "chroma", "test-rag"),
])
def test_clear_removes_only_the_configured_collection(tmp_path, factory):
    with factory(tmp_path) as store:
        chunk = make_chunk("one")
        chunk.content_hash, chunk.embedding_model, chunk.embedding_dimension = "hash", "test", 1
        store.replace_document("document", [chunk], [[1.0]], "signature")
        store.clear()
        assert store.all() == []
        assert store.signature() is None
