# RAG Insight Repository Guidance

## Project objective

Build a compact, evaluated RAG application that demonstrates document ingestion, structure-aware chunking, hybrid retrieval, reranking, bounded corrective retrieval, grounded generation, citations, and measurable results.

Keep the user-facing product limited to:

```text
Upload documents -> Ask a question -> Receive a cited answer
```

## Source of truth

- Read `docs/architecture.md` before changing pipeline behavior.
- Follow `docs/dev-docs/plan-of-action.md` for implementation order and validation gates.
- Use `docs/evaluation.md` for metric definitions and evaluation rules.
- Treat `configs/default.json` and `configs/experiments.json` as reviewable experiment inputs.

## Development commands

Run commands from the repository root with the virtual environment active:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
python -m ruff check .
python -m evaluation.run
streamlit run app.py
```

Use `.\.venv\Scripts\python.exe` directly when activation is unavailable.

## Repository rules

- Target Python 3.11 or newer and keep code compatible with the version declared in `pyproject.toml`.
- Preserve document provenance from ingestion through citations.
- Keep pipeline stages modular and independently testable.
- Do not fabricate benchmark results, confidence values, citations, or model output.
- Keep the original question unchanged for grading and generation; rewritten queries affect retrieval only.
- Permit at most one corrective retrieval retry.
- Keep SQLite exact search as a transparent baseline when adding ChromaDB.
- Add dependencies to `pyproject.toml`; do not add `requirements.txt` unless a deployment target requires it.
- Do not commit `.env`, uploaded documents, generated indexes, model caches, or run artifacts containing document text.
- Never commit or push files under `docs/` or `docs/dev-docs/` unless the user explicitly authorizes those paths in the current request. Keep documentation changes local and uncommitted by default.
- Update documentation when architecture, configuration, commands, or known limitations change.
- Avoid unrelated product features until the core evaluation gates pass.

## Change validation

- Run the smallest relevant tests during development, then the complete test suite before declaring a change complete.
- Run Ruff when Python source or tests change.
- Do not run the held-out evaluation split while tuning retrieval settings.
- Record real experiment configuration and output before making quality claims.

More specific `AGENTS.md` files override or extend this guidance within their directories.
