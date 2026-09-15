import json

import pytest

from rag_insight.config import Settings
from rag_insight.inspection import CharacterBudgetTokenizer, inspect_documents, write_inspection


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)


def test_inspection_records_are_serializable_and_stable(tmp_path):
    document = tmp_path / "guide.md"
    document.write_text("# Guide\n\nUseful evidence.", encoding="utf-8")
    settings = Settings(chunk_strategy="structure", chunk_tokens=80, overlap_tokens=5)
    records = inspect_documents([document], settings, CharacterTokenizer())
    assert records == inspect_documents([document], settings, CharacterTokenizer())
    assert records[0]["strategy"] == "structure"
    assert records[0]["document_version"]
    assert records[0]["tokenizer"] == settings.embedding_model
    assert records[0]["heading_path"] == "Guide"
    assert records[0]["token_count"] == len(records[0]["text"])
    assert records[0]["line_start"] == 1

    output = tmp_path / "chunks.json"
    write_inspection(records, "json", output)
    assert json.loads(output.read_text(encoding="utf-8")) == records


def test_markdown_output_and_invalid_format(tmp_path):
    records = [{"strategy": "recursive", "chunk_id": "id", "document_id": "doc", "document_version": "v",
                "filename": "guide.txt", "heading_path": "", "page": None, "line_start": 1,
                "line_end": 2, "token_count": 4, "text": "text", "tokenizer": "test"}]
    output = tmp_path / "chunks.md"
    write_inspection(records, "markdown", output)
    rendered = output.read_text(encoding="utf-8")
    assert "lines 1-2" in rendered
    assert "Token count: 4" in rendered
    with pytest.raises(ValueError, match="Unsupported inspection format"):
        write_inspection(records, "csv", tmp_path / "chunks.csv")


def test_character_fallback_is_explicit_and_conservative():
    tokenizer = CharacterBudgetTokenizer()
    assert tokenizer.inspection_name.startswith("conservative")
    assert len(tokenizer.encode("café")) == 4
