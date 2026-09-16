import sqlite3
from threading import Thread

import pytest

from rag_insight.models import Chunk
from rag_insight.storage import Store


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
