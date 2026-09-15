"""Persistent SQLite vector store with exact cosine search at demo scale.

Normalized embeddings are stored as JSON. Swap this adapter for an ANN
database when the corpus outgrows an in-memory scan.
"""
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .models import Chunk


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, document TEXT, metadata TEXT, vector TEXT)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)")

    def close(self):
        """Release the database connection owned by this store."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def replace_document(self, document_id, chunks, vectors, signature):
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Each nonempty chunk set must have matching embeddings")
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError("Chunks must belong to the document being replaced")
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
        row = self.connection.execute("SELECT value FROM config WHERE key = 'signature'").fetchone()
        return row[0] if row else None

    def all(self):
        rows = self.connection.execute("SELECT metadata, vector FROM chunks ORDER BY id").fetchall()
        return [(Chunk(**json.loads(meta)), json.loads(vector)) for meta, vector in rows]
