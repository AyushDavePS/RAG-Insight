# Test Suite Guidance

These instructions apply to files under `tests/`.

## Test strategy

- Test observable behavior and important invariants rather than mirroring implementation details.
- Keep unit tests deterministic, offline, and fast.
- Use temporary directories and databases; do not write test data to tracked project directories.
- Use small fakes for tokenizers, embedders, rerankers, and LLMs.
- Do not download models, call Ollama, or connect to Qdrant in ordinary unit tests.
- Mark real-model or external-service checks as integration tests and document their prerequisites.
- Add regression tests for every fixed defect that could silently affect retrieval, grounding, or provenance.

## Required coverage by area

- Ingestion: supported formats, Unicode, empty input, unsupported input, textless PDFs, and source locations.
- Chunking: heading ancestry, fenced code, oversized blocks, overlap, unique IDs, and token limits.
- Storage: round trip, document replacement, transaction behavior, and incompatible index signatures.
- Retrieval: dense ranking, BM25 identifiers, RRF deduplication, empty results, context limits, and MMR diversity.
- Corrective flow: original-question preservation, zero/one retry, regrading, generation gates, and abstention.
- Citations: valid IDs, unknown IDs, missing IDs, duplicate IDs, malformed responses, and source ordering.
- Adapters: contract tests shared by SQLite and Qdrant implementations.

## Commands

Run a focused file while developing:

```powershell
python -m pytest tests/test_retrieval.py
```

Run the complete suite before completion:

```powershell
python -m pytest
python -m ruff check src tests evaluation app.py
```

Tests passing does not establish retrieval quality; benchmark claims belong in the evaluation workflow.
