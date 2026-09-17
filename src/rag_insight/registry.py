"""Deterministic document-level registry derived from persisted chunk evidence."""

import re
from dataclasses import asdict, dataclass

from .metadata import SECTION_PATTERNS, document_profile_role


@dataclass(frozen=True)
class DocumentRecord:
    """A compact, inspectable document projection for planning and aggregation."""

    document_id: str
    filename: str
    version: str
    document_type: str
    profile_role: str
    target_role: str
    sections: tuple[str, ...]
    label: str
    title: str
    summary_chunk_id: str
    skills: tuple[str, ...]

    def trace(self):
        record = asdict(self)
        record["sections"] = list(self.sections)
        record["skills"] = list(self.skills)
        return record


def _first_content_line(chunks):
    for chunk in chunks:
        for line in chunk.text.splitlines():
            value = line.strip()
            if value and not re.fullmatch(r"[A-Z][A-Z /&-]{2,}", value):
                return value[:120]
    return chunks[0].filename


def _title(chunks, label):
    for chunk in chunks:
        lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
        for line in lines[1:5]:
            if (len(line) <= 120 and not re.fullmatch(r"[A-Z][A-Z /&-]{2,}", line)
                    and line != label):
                return line
    return ""


def _skills(chunks):
    values = []
    pattern = re.compile(
        r"(?:CORE|TECHNICAL|KEY)?\s*SKILLS\s*(.*?)(?=\n\s*(?:PROFESSIONAL|WORK|EDUCATION|"
        r"QUALIFICATIONS?|REQUIREMENTS?|RESPONSIBILITIES)\b|$)",
        re.IGNORECASE | re.DOTALL,
    )
    for chunk in chunks:
        match = pattern.search(chunk.text)
        if not match:
            continue
        for value in re.split(r"\||,|\n", match.group(1)):
            cleaned = re.sub(r"\s+", " ", value).strip(" -•\t")
            if 1 < len(cleaned) <= 50 and cleaned.casefold() not in {item.casefold() for item in values}:
                values.append(cleaned)
    return tuple(values[:12])


def _document_type(chunks, sections, profile_role):
    text = "\n".join(chunk.text for chunk in chunks)
    if {"requirements", "responsibilities", "qualifications"} & set(sections):
        return "job_description"
    if profile_role or {"professional_summary", "professional_experience", "education"} & set(sections):
        return "resume_profile"
    if re.search(r"\b(?:invoice|purchase order|bill to)\b", text, re.IGNORECASE):
        return "business_document"
    return "generic_document"


def _sections(chunks):
    return tuple(sorted({
        label
        for chunk in chunks
        for label, pattern in SECTION_PATTERNS
        if pattern.search(chunk.text)
    } | {chunk.section_kind for chunk in chunks if chunk.section_kind}))


def build_document_registry(chunks):
    """Build one evidence-derived record per document without model calls.

    Stored chunks are the durable source of truth. Rebuilding this projection
    after retrieval makes it replacement-safe and upgrades legacy documents
    whenever deterministic metadata extraction improves.
    """
    grouped = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_id, []).append(chunk)
    records = []
    for document_id, group in sorted(grouped.items(), key=lambda item: (item[1][0].filename, item[0])):
        role = document_profile_role(group) or group[0].profile_role
        sections = _sections(group)
        label = _first_content_line(group)
        summary = next((chunk for chunk in group if "professional_summary" in {
            label for label, pattern in SECTION_PATTERNS if pattern.search(chunk.text)
        } or chunk.section_kind == "professional_summary"), None)
        document_type = _document_type(group, sections, role)
        records.append(DocumentRecord(
            document_id=document_id,
            filename=group[0].filename,
            version=group[0].version,
            document_type=document_type,
            profile_role=role if document_type == "resume_profile" else "",
            target_role=role if document_type == "job_description" else "",
            sections=sections,
            label=label,
            title=_title(group, label),
            summary_chunk_id=summary.chunk_id if summary else "",
            skills=_skills(group),
        ))
    return records
