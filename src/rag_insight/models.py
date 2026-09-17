from dataclasses import dataclass, field


@dataclass
class Section:
    document_id: str
    version: str
    filename: str
    text: str
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    heading_path: str = ""
    source_uri: str = ""
    retrieved_at: str = ""
    ocr_model: str = ""


@dataclass
class Chunk(Section):
    chunk_id: str = ""
    content_hash: str = ""
    embedding_model: str = ""
    embedding_dimension: int | None = None
    profile_role: str = ""
    section_kind: str = ""
    chunk_variant: str = ""


@dataclass
class Candidate:
    chunk: Chunk
    score: float


@dataclass
class Grade:
    sufficient: bool
    reason: str


@dataclass
class Answer:
    text: str
    sources: list[Chunk]
    sufficient: bool
    reason: str
    trace: list[dict] = field(default_factory=list)
