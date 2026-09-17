"""Paragraph-recursive baseline and heading/code-aware chunking.

Token budgets use the embedding tokenizer, including heading prefixes.
Oversized blocks fall back to overlapping line/character windows.
"""
import re
from dataclasses import asdict, replace
from hashlib import sha256

from .models import Chunk, Section


def _blocks(section: Section, structure: bool) -> list[Section]:
    lines = section.text.splitlines(keepends=True)
    headings: list[tuple[int, str]] = []
    output = []
    buffer = []
    start = 0
    fence = None

    def flush(end):
        if buffer and "".join(buffer).strip():
            output.append(replace(
                section, text="".join(buffer).strip(),
                heading_path=" > ".join(title for _, title in headings),
                line_start=(section.line_start + start) if section.line_start else None,
                line_end=(section.line_start + end - 1) if section.line_start else None,
            ))
        buffer.clear()

    for i, line in enumerate(lines):
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line) if structure and not fence else None
        marker = re.match(r"^\s*(`{3,}|~{3,})", line) if structure else None
        if heading:
            flush(i)
            depth, title = len(heading[1]), heading[2]
            headings = [(d, t) for d, t in headings if d < depth] + [(depth, title)]
        if not buffer:
            start = i
        buffer.append(line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        if not line.strip() and not fence:
            flush(i + 1)
    flush(len(lines))
    return output


def _chunk_sections(sections, settings, tokenizer, structure, variant) -> list[Chunk]:
    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    chunks = []
    for section in sections:
        for block in _blocks(section, structure):
            prefix = f"{block.heading_path}\n\n" if block.heading_path else ""
            # Avoid a very long heading consuming the complete budget.
            if count(prefix) >= settings.chunk_tokens // 2:
                prefix = ""
            position = 0
            while position < len(block.text):
                low, high = position + 1, len(block.text)
                end = position
                while low <= high:
                    mid = (low + high) // 2
                    if count(prefix + block.text[position:mid]) <= settings.chunk_tokens:
                        end, low = mid, mid + 1
                    else:
                        high = mid - 1
                if end == position:
                    raise ValueError("Chunk budget cannot fit a character; increase chunk_tokens")
                # Prefer a line boundary when splitting an oversized block.
                if end < len(block.text):
                    boundary = block.text.rfind("\n", position, end)
                    if boundary > position + (end - position) // 2:
                        end = boundary + 1
                text = prefix + block.text[position:end]
                meta = asdict(block)
                meta["text"] = text
                if block.line_start is not None:
                    meta["line_start"] = block.line_start + block.text[:position].count("\n")
                    meta["line_end"] = block.line_start + block.text[:end].rstrip("\n").count("\n")
                content_hash = sha256(text.encode("utf-8")).hexdigest()
                identity = f"{block.document_id}:{block.version}:{len(chunks)}:{content_hash}"
                chunks.append(Chunk(
                    **meta,
                    chunk_id=sha256(identity.encode()).hexdigest()[:20],
                    content_hash=content_hash,
                    chunk_variant=variant,
                ))
                if end == len(block.text):
                    break
                next_start = end
                # Prefixes are embedding-only context and must not consume the
                # body overlap budget or distort source locations.
                while next_start > position + 1 and count(block.text[next_start - 1:end]) <= settings.overlap_tokens:
                    next_start -= 1
                position = next_start
    return chunks


def chunk_sections(sections, settings, tokenizer) -> list[Chunk]:
    """Build recursive, structure-aware, or deduplicated hybrid chunk variants.

    Hybrid retains structure-aware chunks first, then adds recursive chunks only
    when their exact text is not already represented. This preserves heading-rich
    retrieval while retaining a heading-free fallback for the same source.
    """
    if settings.chunk_strategy == "structure":
        return _chunk_sections(sections, settings, tokenizer, structure=True, variant="structure")
    if settings.chunk_strategy == "recursive":
        return _chunk_sections(sections, settings, tokenizer, structure=False, variant="recursive")
    structured = _chunk_sections(sections, settings, tokenizer, structure=True, variant="structure")
    recursive = _chunk_sections(sections, settings, tokenizer, structure=False, variant="recursive")
    output, seen = [], set()
    for chunk in [*structured, *recursive]:
        # Exact duplicates are common for flat PDFs/text files; retaining both
        # would only waste candidate and context capacity.
        key = (chunk.document_id, chunk.page, chunk.line_start, chunk.line_end, chunk.content_hash)
        if key not in seen:
            output.append(chunk)
            seen.add(key)
    return output
