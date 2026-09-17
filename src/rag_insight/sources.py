"""Optional, local-only source adapters using the standard ingestion contract."""
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path


@dataclass(frozen=True)
class SourceDocument:
    uri: str
    title: str
    text: str
    retrieved_at: str
    source_hash: str


class SourceAdapter:
    def fetch(self, request) -> list[SourceDocument]:
        raise NotImplementedError


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title, self.parts, self._in_title, self._ignored = "", [], False, 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._ignored += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self._ignored = max(0, self._ignored - 1)
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        value = " ".join(data.split())
        if not value or self._ignored:
            return
        if self._in_title:
            self.title += value
        else:
            self.parts.append(value)


class LocalHtmlAdapter(SourceAdapter):
    """Fixture adapter; deliberately rejects HTTP(S) and keeps source fetching local."""

    def fetch(self, request) -> list[SourceDocument]:
        path = Path(request)
        if path.suffix.lower() not in {".html", ".htm"}:
            raise ValueError("Local HTML adapter supports only .html or .htm files")
        if not path.is_file():
            raise ValueError(f"Local HTML source was not found: {path}")
        raw = path.read_bytes()
        parser = _TextExtractor()
        parser.feed(raw.decode("utf-8-sig"))
        text = "\n\n".join(parser.parts).strip()
        if not text:
            raise ValueError(f"{path.name}: source contains no extractable HTML text")
        return [SourceDocument(
            uri=path.resolve().as_uri(),
            title=parser.title.strip() or path.stem,
            text=text,
            retrieved_at=datetime.now(UTC).isoformat(),
            source_hash=sha256(raw).hexdigest(),
        )]
