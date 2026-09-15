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


@dataclass
class Chunk(Section):
    chunk_id: str = ""


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
