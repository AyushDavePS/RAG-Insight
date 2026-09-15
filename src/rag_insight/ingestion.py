"""Parse UTF-8 text and text-based PDFs; preserve source locations."""
from hashlib import sha256
from pathlib import Path

from .models import Section


def parse_document(path: Path) -> list[Section]:
    raw = path.read_bytes()
    version = sha256(raw).hexdigest()
    # Filenames are collection-local identities; reuploading replaces that document.
    document_id = sha256(path.name.encode()).hexdigest()[:16]
    metadata = {"document_id": document_id, "version": version, "filename": path.name}
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        sections = [
            Section(**metadata, text=page.extract_text() or "", page=i + 1)
            for i, page in enumerate(PdfReader(path).pages)
        ]
        if not any(section.text.strip() for section in sections):
            raise ValueError(f"{path.name}: no extractable text; OCR is not supported")
        return [section for section in sections if section.text.strip()]
    if path.suffix.lower() not in {".md", ".txt"}:
        raise ValueError(f"Unsupported document type: {path.suffix}")
    text = raw.decode("utf-8-sig")
    if not text.strip():
        raise ValueError(f"{path.name}: document is empty")
    return [Section(**metadata, text=text, line_start=1, line_end=len(text.splitlines()))]
