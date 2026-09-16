"""Reproducible trace generation for the Phase 2 dense baseline."""
import json
from dataclasses import asdict
from pathlib import Path

SMOKE_QUERIES = (
    "Why was Redis selected?",
    "Which header authenticates API requests?",
    "What happens after ingestion retries fail?",
    "Which AWS region hosts the service?",
)


def run_baseline(pipeline, queries=SMOKE_QUERIES, expected_evidence=None):
    """Retrieve queries and return a serializable baseline trace."""
    settings = pipeline.settings
    records = []
    for question in queries:
        context, trace = pipeline.retrieve(question)
        selected = [asdict(candidate.chunk) for candidate in context]
        labels = expected_evidence.get(question) if expected_evidence else None
        evidence_selected = None if labels is None else [
            {"filename": label["filename"], "quote": label["quote"],
             "selected": any(label["filename"] == chunk["filename"] and label["quote"] in chunk["text"]
                             for chunk in selected)}
            for label in labels
        ]
        records.append({
            "question": question,
            "trace": trace,
            "selected_context": selected,
            "selected_context_ids": [candidate.chunk.chunk_id for candidate in context],
            "evidence_labels_selected": evidence_selected,
        })
    return {
        "stage": f"{settings.chunk_strategy}_dense",
        "settings": asdict(settings),
        "embedding": {
            "backend": settings.embedding_backend,
            "model": settings.embedding_model,
            "dimension": next((record["trace"].get("embedding_dimension") for record in records
                                if record["trace"].get("embedding_dimension") is not None), None),
        },
        "records": records,
    }


def write_baseline(result, output):
    """Write a baseline trace as UTF-8 JSON."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
