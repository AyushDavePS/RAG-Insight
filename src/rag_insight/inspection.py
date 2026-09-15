"""Offline chunk inspection without embeddings, indexes, or model calls."""
import json
from dataclasses import asdict
from pathlib import Path

from .chunking import chunk_sections
from .ingestion import parse_document


class CharacterBudgetTokenizer:
    """Dependency-free conservative fallback for offline inspection."""

    inspection_name = "conservative character fallback (not embedding tokens)"

    def encode(self, text, **kwargs):
        return list(text)


def inspect_documents(paths, settings, tokenizer):
    """Parse and chunk files into serializable provenance records."""
    records = []
    for path in sorted((Path(path) for path in paths), key=lambda item: item.name.lower()):
        chunks = chunk_sections(parse_document(path), settings, tokenizer)
        for chunk in chunks:
            record = asdict(chunk)
            record["document_version"] = record.pop("version")
            record["strategy"] = settings.chunk_strategy
            record["tokenizer"] = getattr(tokenizer, "inspection_name", settings.embedding_model)
            record["token_count"] = len(tokenizer.encode(chunk.text, add_special_tokens=False))
            records.append(record)
    return records


def _location(record):
    if record["page"] is not None:
        return f"page {record['page']}"
    if record["line_start"] is not None:
        return f"lines {record['line_start']}-{record['line_end']}"
    return "source location unavailable"


def format_inspection(records, output_format):
    """Format inspection records as JSON or readable Markdown."""
    if output_format == "json":
        return json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    elif output_format == "markdown":
        parts = ["# Chunk inspection\n"]
        for record in records:
            heading = record["heading_path"] or "(none)"
            parts.append(
                f"## {record['filename']} — {record['chunk_id']}\n\n"
                f"- Strategy: `{record['strategy']}`\n"
                f"- Document ID: `{record['document_id']}`\n"
                f"- Version: `{record['document_version']}`\n"
                f"- Heading path: {heading}\n"
                f"- Location: {_location(record)}\n"
                f"- Tokenizer: {record['tokenizer']}\n"
                f"- Token count: {record['token_count']}\n\n"
                f"```text\n{record['text']}\n```\n"
            )
        return "\n".join(parts)
    else:
        raise ValueError("Unsupported inspection format; use 'json' or 'markdown'")


def write_inspection(records, output_format, output):
    """Write formatted inspection records to a local file."""
    output = Path(output)
    content = format_inspection(records, output_format)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
