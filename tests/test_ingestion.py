from pathlib import Path

import pytest

from rag_insight.ingestion import parse_document


def test_markdown_and_unicode_text_preserve_content_and_locations(tmp_path):
    markdown = tmp_path / "guide.md"
    markdown.write_text("# Heading\n\nA café.\n", encoding="utf-8")
    unicode_file = tmp_path / "unicode.txt"
    unicode_file.write_bytes("\ufeffनमस्ते\n".encode())

    section = parse_document(markdown)[0]
    assert section.text == markdown.read_bytes().decode("utf-8")
    assert (section.line_start, section.line_end) == (1, 3)
    assert parse_document(unicode_file)[0].text == "नमस्ते\n"


def test_empty_and_unsupported_documents_fail_actionably(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text(" \n\t", encoding="utf-8")
    unsupported = tmp_path / "notes.docx"
    unsupported.write_text("not a document", encoding="utf-8")

    with pytest.raises(ValueError, match="empty.txt: document is empty"):
        parse_document(empty)
    with pytest.raises(ValueError, match="Unsupported document type: .docx"):
        parse_document(unsupported)


def test_identity_uses_filename_and_version_uses_content(tmp_path):
    first = tmp_path / "same.txt"
    first.write_text("first", encoding="utf-8")
    changed = parse_document(first)[0]
    first.write_text("second", encoding="utf-8")
    second = parse_document(first)[0]
    renamed = tmp_path / "renamed.txt"
    renamed.write_text("second", encoding="utf-8")
    third = parse_document(renamed)[0]

    assert changed.document_id == second.document_id
    assert changed.version != second.version
    assert second.document_id != third.document_id
    assert second.version == third.version


def test_pdf_preserves_original_nonempty_page_numbers(monkeypatch, tmp_path):
    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class Reader:
        def __init__(self, path):
            self.pages = [Page("first page"), Page(" "), Page("third page")]

    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", Reader)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    sections = parse_document(source)
    assert [(section.text, section.page) for section in sections] == [("first page", 1), ("third page", 3)]


def test_textless_pdf_fails_explicitly(monkeypatch, tmp_path):
    class Reader:
        def __init__(self, path):
            self.pages = [type("Page", (), {"extract_text": lambda self: ""})()]

    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", Reader)
    (tmp_path / "scan.pdf").write_bytes(b"pdf")
    with pytest.raises(ValueError, match="no extractable text; OCR is not supported"):
        parse_document(Path(tmp_path / "scan.pdf"))
