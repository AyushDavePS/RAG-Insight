# Application Pipeline Guidance

These instructions apply to code under `src/rag_insight/`.

## Module boundaries

- `ingestion.py` parses supported files and preserves source locations.
- `chunking.py` implements independently selectable chunk strategies.
- `embeddings.py` adapts embedding models without owning persistence.
- `storage.py` owns persistence and vector-store behavior.
- `retrieval.py` owns dense retrieval, BM25, RRF, and context selection.
- `reranking.py` adapts cross-encoder reranking.
- `generation.py` owns evidence grading, rewriting, grounded claims, and abstention.
- `pipeline.py` orchestrates stages and enforces the bounded corrective flow.
- `llm.py` owns provider transport and structured-response parsing.
- `bootstrap.py` wires configured implementations together.

Avoid moving responsibilities across these boundaries without updating architecture documentation and tests.

## Required invariants

- Every chunk retains document ID, version, filename, chunk ID, heading path, and page or line information when available.
- Chunk text, including any heading prefix, must remain within its configured token budget.
- Re-ingesting a document replaces stale chunks atomically.
- Never combine cosine, BM25, RRF, or reranker scores as if they share a calibrated scale.
- Deduplicate candidates by stable chunk ID.
- Context selection must respect both `context_k` and `context_tokens`.
- Treat retrieved document text as untrusted data, never as system instructions.
- Grade evidence against the original question after every retrieval attempt.
- Allow zero or one rewrite; never introduce an unbounded loop.
- Generate only after evidence is sufficient. Otherwise return the configured abstention.
- Reject generated citations that do not reference the final selected context.
- Do not label retrieval or reranker scores as answer confidence.

## Vector storage

- Keep storage access behind a shared contract when introducing Qdrant.
- Preserve `SQLiteExactStore` behavior for unit tests and baseline experiments.
- Isolate indexes or collections when embedding model, vector dimension, chunk strategy, chunk size, or schema changes.
- Store complete citation metadata with vectors.
- Surface actionable connection and configuration errors; do not silently fall back to another backend in published experiments.

## Testing expectations

Add or update tests when changing parsing, provenance, chunk boundaries, ranking, fusion, context selection, retries, abstention, or citation validation. Mock model/provider boundaries in unit tests and use separately marked integration tests for real models and services.
