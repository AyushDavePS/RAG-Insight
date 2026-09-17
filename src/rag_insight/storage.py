"""Interchangeable local vector stores for the RAG pipeline."""
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import Protocol

from .models import Chunk


class VectorStore(Protocol):
    def replace_document(self, document_id, chunks, vectors, signature): ...
    def signature(self): ...
    def all(self): ...
    def cached_vectors(self, content_hashes, embedding_model): ...
    def dense_search(self, query_vector, k): ...
    def close(self): ...
    def clear(self): ...


def _validate_replacement(document_id, chunks, vectors):
    if not chunks or len(chunks) != len(vectors):
        raise ValueError("Each nonempty chunk set must have matching embeddings")
    dimensions = {len(vector) for vector in vectors}
    if not dimensions or 0 in dimensions or len(dimensions) != 1:
        raise ValueError("All stored embeddings must have one nonzero dimension")
    if any(chunk.document_id != document_id for chunk in chunks):
        raise ValueError("Chunks must belong to the document being replaced")


class SQLiteExactStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Streamlit may rerun a session on a different script thread. SQLite
        # operations remain serialized by the re-entrant lock below.
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = RLock()
        self.connection.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, document TEXT, metadata TEXT, vector TEXT)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)")

    def close(self):
        """Release the database connection owned by this store."""
        with self._lock:
            if self.connection is not None:
                self.connection.close()
                self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def replace_document(self, document_id, chunks, vectors, signature):
        _validate_replacement(document_id, chunks, vectors)
        with self._lock:
            previous = self.signature()
            if previous is not None and previous != signature:
                raise ValueError("Index configuration differs. Use a separate index for this experiment.")
            with self.connection:
                self.connection.execute("INSERT OR REPLACE INTO config VALUES ('signature', ?)", (signature,))
                self.connection.execute("DELETE FROM chunks WHERE document = ?", (document_id,))
                self.connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", [
                    (chunk.chunk_id, document_id, json.dumps(asdict(chunk)), json.dumps(vector))
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ])

    def signature(self):
        with self._lock:
            row = self.connection.execute("SELECT value FROM config WHERE key = 'signature'").fetchone()
            return row[0] if row else None

    def all(self):
        with self._lock:
            rows = self.connection.execute("SELECT metadata, vector FROM chunks ORDER BY id").fetchall()
            return [(Chunk(**json.loads(meta)), json.loads(vector)) for meta, vector in rows]

    def cached_vectors(self, content_hashes, embedding_model):
        """Return reusable vectors by content hash for the active compatible index."""
        wanted = set(content_hashes)
        if not wanted:
            return {}
        with self._lock:
            rows = self.connection.execute("SELECT metadata, vector FROM chunks").fetchall()
        cached = {}
        for metadata, vector in rows:
            chunk = Chunk(**json.loads(metadata))
            if (chunk.content_hash in wanted and chunk.embedding_model == embedding_model
                    and chunk.embedding_dimension == len(json.loads(vector))):
                cached.setdefault(chunk.content_hash, json.loads(vector))
        return cached

    def dense_search(self, query_vector, k):
        from .retrieval import dense_search

        return dense_search(query_vector, self.all(), k)

    def clear(self):
        with self._lock, self.connection:
            self.connection.execute("DELETE FROM chunks")
            self.connection.execute("DELETE FROM config")


class ChromaStore:
    """Persistent Chroma adapter retaining complete chunk provenance."""

    def __init__(self, path: Path, collection_name: str):
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError(
                "ChromaDB is not installed. Install project dependencies before using vector_backend='chroma'."
            ) from error
        try:
            path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(path))
            self.collection = self.client.get_or_create_collection(
                collection_name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as error:
            raise RuntimeError(f"Could not initialize ChromaDB at {path}: {error}") from error

    def close(self):
        # PersistentClient has no required close operation.
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def signature(self):
        return (self.collection.metadata or {}).get("rag_insight_signature") or None

    def _validate_signature(self, signature):
        previous = self.signature()
        if previous is not None and previous != signature:
            raise ValueError("Index configuration differs. Use a separate index for this experiment.")

    def replace_document(self, document_id, chunks, vectors, signature):
        _validate_replacement(document_id, chunks, vectors)
        self._validate_signature(signature)
        ids = [chunk.chunk_id for chunk in chunks]
        metadata = [
            {
                "document_id": chunk.document_id,
                "content_hash": chunk.content_hash,
                "embedding_model": chunk.embedding_model,
                "embedding_dimension": chunk.embedding_dimension or len(vector),
                "chunk": json.dumps(asdict(chunk)),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            # Upsert first means a failed request never removes existing document evidence.
            self.collection.upsert(ids=ids, documents=[chunk.text for chunk in chunks],
                                   embeddings=vectors, metadatas=metadata)
            stale = self.collection.get(where={"document_id": document_id}, include=[])["ids"]
            stale = [chunk_id for chunk_id in stale if chunk_id not in set(ids)]
            if stale:
                self.collection.delete(ids=stale)
            actual = set(self.collection.get(where={"document_id": document_id}, include=[])["ids"])
            if actual != set(ids):
                raise RuntimeError("ChromaDB replacement verification failed; the document may need re-indexing.")
            if self.signature() is None:
                # Chroma rejects ``hnsw:space`` in modify calls because the
                # distance function is immutable after collection creation.
                self.collection.modify(metadata={"rag_insight_signature": signature})
        except Exception as error:
            raise RuntimeError(f"ChromaDB document replacement failed: {error}") from error

    def all(self):
        try:
            data = self.collection.get(include=["metadatas", "embeddings"])
        except Exception as error:
            raise RuntimeError(f"Could not read ChromaDB collection: {error}") from error
        rows = [
            (Chunk(**json.loads(metadata["chunk"])), [float(value) for value in vector])
            for metadata, vector in zip(data["metadatas"], data["embeddings"], strict=True)
        ]
        return sorted(rows, key=lambda row: row[0].chunk_id)

    def cached_vectors(self, content_hashes, embedding_model):
        wanted = set(content_hashes)
        if not wanted:
            return {}
        cached = {}
        for chunk, vector in self.all():
            if chunk.content_hash in wanted and chunk.embedding_model == embedding_model:
                cached.setdefault(chunk.content_hash, vector)
        return cached

    def dense_search(self, query_vector, k):
        from .models import Candidate

        if not self.all():
            return []
        try:
            data = self.collection.query(query_embeddings=[query_vector], n_results=k,
                                         include=["metadatas", "distances"])
        except Exception as error:
            raise RuntimeError(f"ChromaDB dense search failed: {error}") from error
        return [
            Candidate(Chunk(**json.loads(metadata["chunk"])), 1 - distance)
            for metadata, distance in zip(data["metadatas"][0], data["distances"][0], strict=True)
        ]

    def clear(self):
        """Remove only records in this configured collection."""
        ids = self.collection.get(include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)
        self.collection.modify(metadata={"rag_insight_signature": ""})


# Backwards-compatible name for the transparent exact-search baseline.
Store = SQLiteExactStore
