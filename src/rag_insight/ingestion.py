"""Parse UTF-8 text and text-based PDFs; preserve source locations."""
from hashlib import sha256
from pathlib import Path

from .models import Section
from .sources import SourceDocument


def parse_document(path: Path, ocr_extractor=None) -> list[Section]:
    raw = path.read_bytes()
    version = sha256(raw).hexdigest()
    # Filenames are collection-local identities; reuploading replaces that document.
    document_id = sha256(path.name.encode()).hexdigest()[:16]
    metadata = {
        "document_id": document_id,
        "version": version,
        "filename": path.name,
        "source_uri": path.resolve().as_uri(),
    }
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        sections = [
            Section(**metadata, text=page.extract_text() or "", page=i + 1)
            for i, page in enumerate(PdfReader(path).pages)
        ]
        if not any(section.text.strip() for section in sections):
            if ocr_extractor is not None:
                ocr_pages = ocr_extractor.extract_pdf(path)
                if len(ocr_pages) != len(sections):
                    raise RuntimeError("OCR did not return one result per PDF page")
                extracted = [
                    Section(**metadata, text=text, page=index + 1, ocr_model=ocr_extractor.model_id)
                    for index, text in enumerate(ocr_pages)
                    if text.strip()
                ]
                if extracted:
                    return extracted
                raise ValueError(f"{path.name}: OCR found no readable text")
            raise ValueError(f"{path.name}: no extractable text; OCR is not supported")
        return [section for section in sections if section.text.strip()]
    if path.suffix.lower() not in {".md", ".txt"}:
        raise ValueError(f"Unsupported document type: {path.suffix}")
    text = raw.decode("utf-8-sig")
    if not text.strip():
        raise ValueError(f"{path.name}: document is empty")
    return [Section(**metadata, text=text, line_start=1, line_end=len(text.splitlines()))]


def parse_source(source: SourceDocument) -> list[Section]:
    """Convert an optional adapter result into the same provenance-preserving sections."""
    if not source.text.strip():
        raise ValueError(f"{source.title}: source is empty")
    filename = source.title.strip() or "source"
    document_id = sha256(source.uri.encode()).hexdigest()[:16]
    return [Section(
        document_id=document_id,
        version=source.source_hash,
        filename=filename,
        text=source.text,
        line_start=1,
        line_end=len(source.text.splitlines()),
        source_uri=source.uri,
        retrieved_at=source.retrieved_at,
    )]
