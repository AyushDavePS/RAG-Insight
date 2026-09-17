"""Deterministic document metadata used for safe local retrieval filters."""

import re
from dataclasses import replace

ROLE_PATTERNS = (
    ("business_development", re.compile(r"\b(?:bde|business development)(?:\s*/\s*sales)?\b", re.IGNORECASE)),
    ("frontend_development", re.compile(r"\b(?:front[ -]?end|frontend) developer\b", re.IGNORECASE)),
    # Keep compound specialisations ahead of their broader AI/ML counterpart.
    # These patterns are title-shaped, so a document merely mentioning AI/ML in
    # its skills or projects does not become a role-filter match.
    ("fullstack_aiml", re.compile(
        r"\b(?:full[ -]?stack|fullstack)\b[^\n]{0,60}"
        r"\b(?:ai\s*/\s*ml|aiml|artificial intelligence|machine learning)\b|"
        r"\b(?:ai\s*/\s*ml|aiml|artificial intelligence|machine learning)\b[^\n]{0,60}"
        r"\b(?:full[ -]?stack|fullstack)\b",
        re.IGNORECASE,
    )),
    ("aiml", re.compile(
        r"\b(?:ai\s*/\s*ml|aiml|artificial intelligence|machine learning)\s+"
        r"(?:engineer|developer|scientist|specialist)\b",
        re.IGNORECASE,
    )),
)
ROLE_KEYWORDS = {"analyst", "architect", "consultant", "designer", "developer", "engineer", "executive",
                 "manager", "scientist", "specialist"}

# These are generic document sections, not document-type classifications.  They
# are deliberately conservative: a heading-like label earns metadata, while a
# word occurring in prose does not.
SECTION_PATTERNS = (
    ("professional_summary", re.compile(r"\bPROFESSIONAL\s+SUMMARY\b", re.IGNORECASE)),
    ("professional_experience", re.compile(
        r"\b(?:PROFESSIONAL|WORK)\s+EXPERIENCE\b|\bEMPLOYMENT\s+HISTORY\b", re.IGNORECASE
    )),
    ("education", re.compile(r"\b(?:EDUCATION|ACADEMIC\s+(?:BACKGROUND|QUALIFICATIONS?))\b", re.IGNORECASE)),
    ("qualifications", re.compile(r"\b(?:REQUIRED|PREFERRED|MINIMUM)?\s*QUALIFICATIONS\b", re.IGNORECASE)),
    ("requirements", re.compile(r"\b(?:JOB|ROLE|POSITION|TECHNICAL)?\s*REQUIREMENTS\b", re.IGNORECASE)),
    ("responsibilities", re.compile(r"\b(?:KEY|JOB|ROLE)?\s*RESPONSIBILITIES\b", re.IGNORECASE)),
    ("skills", re.compile(r"\b(?:CORE|TECHNICAL|KEY)?\s*SKILLS\b", re.IGNORECASE)),
    ("benefits", re.compile(r"\b(?:BENEFITS|WHAT\s+WE\s+OFFER)\b", re.IGNORECASE)),
)


def profile_role(text):
    """Return a conservative normalized role label, or an empty label when unknown."""
    title_region = "\n".join(text.splitlines()[:8])
    for label, pattern in ROLE_PATTERNS:
        if pattern.search(title_region):
            return label
    for line in text.splitlines()[:8]:
        normalized = re.sub(r"[^a-z0-9]+", " ", line.lower()).strip()
        words = normalized.split()
        if 1 < len(words) <= 6 and ROLE_KEYWORDS & set(words):
            return "_".join(words)
    return ""


def document_profile_role(chunks):
    """Find a document role even when persisted chunks are not source ordered."""
    # SQLite and Chroma return chunk IDs in storage order, not page/line order.
    # Test each chunk independently so the title/summary chunk is not lost when
    # a later education or work-history chunk happens to sort first.
    for label, pattern in ROLE_PATTERNS:
        if any(pattern.search(chunk.text) for chunk in chunks):
            return label
    for chunk in chunks:
        role = profile_role(chunk.text)
        if role:
            return role
    return ""


def section_kind(text):
    for label, pattern in SECTION_PATTERNS:
        if pattern.search(text):
            return label
    return ""


def enrich_chunks(chunks):
    """Attach document-wide role and chunk-local section metadata without model calls."""
    document_text = {}
    for chunk in chunks:
        document_text.setdefault(chunk.document_id, []).append(chunk)
    document_roles = {document_id: document_profile_role(chunks)
                      for document_id, chunks in document_text.items()}
    return [replace(chunk, profile_role=document_roles[chunk.document_id] or chunk.profile_role,
                    section_kind=section_kind(chunk.text)) for chunk in chunks]
