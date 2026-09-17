import math
import re
from collections import Counter

from .models import Candidate


def terms(text):
    # Intent and metadata matching are deliberately case-insensitive.  Keep
    # the original user question for generation and citations; only control
    # decisions use this normalized representation.
    return re.findall(r"[\w]+(?:[-.][\w]+)*", text.casefold())


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def dense_search(query_vector, rows, k):
    dimensions = {len(vector) for _, vector in rows}
    if len(dimensions) > 1 or (dimensions and len(query_vector) not in dimensions):
        raise ValueError("Query and stored embedding dimensions do not match")
    return sorted([Candidate(chunk, dot(query_vector, vector)) for chunk, vector in rows],
                  key=lambda item: item.score, reverse=True)[:k]


def bm25_search(query, rows, k):
    if not rows:
        return []
    counters = [Counter(terms(chunk.text)) for chunk, _ in rows]
    lengths = [sum(counter.values()) for counter in counters]
    average = sum(lengths) / len(lengths) or 1
    frequency = Counter(term for counter in counters for term in counter)
    results = []
    for (chunk, _), counter, length in zip(rows, counters, lengths, strict=True):
        score = 0.0
        for term in set(terms(query)):
            tf = counter[term]
            idf = math.log(1 + (len(rows) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / average))
        if score > 0:
            results.append(Candidate(chunk, score))
    return sorted(results, key=lambda item: item.score, reverse=True)[:k]


def fuse(rankings, k=60):
    scores, chunks = {}, {}
    for ranking in rankings:
        for rank, candidate in enumerate(ranking, 1):
            key = candidate.chunk.chunk_id
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            chunks[key] = candidate.chunk
    return sorted([Candidate(chunks[key], score) for key, score in scores.items()],
                  key=lambda item: item.score, reverse=True)


def _focus_terms(query):
    stop_words = {"a", "all", "an", "and", "are", "at", "for", "from", "how", "in", "inside", "is", "list",
                  "of", "on", "out", "present", "the", "their", "to", "what", "when", "where", "which", "who",
                  "why", "with"}
    return {_normalize_term(term) for term in terms(query)
            if term not in stop_words}


def _normalize_term(term):
    """Apply only safe plural normalization; never turn Paris into ``pari``."""
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 3 and term.endswith("s") and not term.endswith(("is", "ss", "us")):
        return term[:-1]
    return term


def _term_overlap(query_terms, text):
    text_terms = {_normalize_term(term) for term in terms(text)}
    return len(query_terms & text_terms)


def _is_aggregate_query(query):
    query_terms = set(terms(query))
    return bool({"all", "each", "every", "list", "compare", "both", "across", "other", "remaining"} & query_terms
                or "summary" in query_terms and any(term.endswith("s") for term in query_terms))


def requests_role_differentiation(query):
    """Recognize an explicit request to contrast role or document groups."""
    query_terms = set(terms(query))
    return bool({"differentiate", "difference", "between", "versus", "vs"} & query_terms)


def requires_document_coverage(query, semantic_intent=None):
    """Recognize collection-wide requests that need evidence from each document."""
    query_terms = set(terms(query))
    summary_intent = {"summarize", "summary", "summaries", "overview", "overviews", "profile", "profiles"}
    comparative_intent = {"best", "better", "compare", "comparison", "rank", "ranking", "recommend", "strongest",
                          "highest", "lowest", "most", "least", "maximum", "minimum", "top", "bottom"}
    listing_intent = {"all", "each", "every", "list", "across", "companies", "employers", "skills",
                      "requirements", "responsibilities", "qualifications"}
    collection_scope = {"given", "document", "documents", "file", "files", "resume", "resumes",
                        "candidate", "candidates", "bde", "bdes", "other", "remaining"}
    heuristic = bool((summary_intent | comparative_intent | listing_intent) & query_terms
                     and collection_scope & query_terms)
    return heuristic or requests_role_differentiation(query) or bool((semantic_intent or {}).get("collection_wide")
                             or (semantic_intent or {}).get("aggregation"))


def is_structured_aggregation_request(query):
    """Recognize collection-wide questions that need a fact from each document.

    This only broadens retrieval coverage. It does not choose a reducer or
    change the answer: those remain explicit, evidence-backed controller
    decisions.
    """
    query_terms = set(terms(query))
    operations = {"count", "counts", "how", "many", "total", "highest", "lowest", "most", "least",
                  "maximum", "minimum", "more", "less", "over", "under", "filter", "matching"}
    scope = {"candidate", "candidates", "resume", "resumes", "profile", "profiles", "document", "documents",
             "bde", "bdes"}
    return bool(operations & query_terms and scope & query_terms)


def is_comparative_query(query, semantic_intent=None):
    query_terms = set(terms(query))
    comparative_intent = {"best", "better", "compare", "comparison", "rank", "ranking", "recommend", "strongest",
                          "highest", "most", "maximum", "top"}
    collection_scope = {"candidate", "candidates", "resume", "resumes", "profile", "profiles", "document", "documents"}
    heuristic = bool(comparative_intent & query_terms and collection_scope & query_terms)
    return heuristic or bool((semantic_intent or {}).get("comparative"))


def requested_profile_roles(query, available_roles=()):
    """Map explicit user role terms to stored normalized role metadata."""
    query_terms = set(terms(query))
    normalized_query = query.casefold()
    roles = set()
    if {"bde", "bdes"} & query_terms or {"business", "development"} <= query_terms:
        roles.add("business_development")
    if {"frontend", "front-end"} & query_terms:
        roles.add("frontend_development")
    aiml_phrase = r"(?:ai\s*/\s*ml|aiml|machine\s+learning|artificial\s+intelligence)"
    fullstack_phrase = r"(?:full[ -]?stack|fullstack)"
    compound_aiml = re.compile(
        rf"\b{fullstack_phrase}\b[^\n]{{0,40}}?\b{aiml_phrase}\b|"
        rf"\b{aiml_phrase}\b[^\n]{{0,40}}?\b{fullstack_phrase}\b",
        re.IGNORECASE,
    )
    compound_matches = list(compound_aiml.finditer(normalized_query))
    without_compound_aiml = compound_aiml.sub(" ", normalized_query)
    if re.search(aiml_phrase, without_compound_aiml, re.IGNORECASE) or (not compound_matches and re.search(
            aiml_phrase, normalized_query, re.IGNORECASE)):
        roles.add("aiml")
    if compound_matches:
        roles.add("fullstack_aiml")
    for role in set(available_roles):
        role_terms = set(role.split("_"))
        if len(role_terms) > 1 and role_terms <= query_terms:
            roles.add(role)
    return roles


def filter_rows_by_profile(rows, query):
    requested = requested_profile_roles(query, (chunk.profile_role for chunk, _ in rows))
    if not requested:
        return list(rows), requested
    return [(chunk, vector) for chunk, vector in rows if chunk.profile_role in requested], requested


def prefers_professional_summary(query):
    query_terms = set(terms(query))
    return bool({"summary", "summaries", "profile", "profiles"} & query_terms)


def preferred_section_kinds(query):
    """Return generic section concepts that may receive a soft context boost."""
    query_terms = set(terms(query))
    preferred = set()
    if {"education", "educational", "degree", "studied", "study", "academic"} & query_terms:
        preferred.add("education")
    if {"qualification", "qualifications", "certification", "certifications"} & query_terms:
        preferred.update({"education", "qualifications"})
    if {"requirement", "requirements", "required"} & query_terms:
        preferred.update({"requirements", "qualifications"})
    if {"responsibility", "responsibilities", "duties"} & query_terms:
        preferred.add("responsibilities")
    if {"skill", "skills", "competency", "competencies"} & query_terms:
        preferred.add("skills")
    if {"benefit", "benefits", "perks"} & query_terms:
        preferred.add("benefits")
    if {"company", "companies", "employer", "employers", "worked", "work", "employment", "tenure"} & query_terms:
        preferred.add("professional_experience")
    return preferred


def explicitly_matched_document_ids(query, rows):
    """Find uniquely named documents from query terms without filtering others out."""
    intent_terms = {"education", "educational", "degree", "studied", "study", "academic",
                    "qualification", "qualifications", "certification", "certifications", "requirement",
                    "requirements", "required", "responsibility", "responsibilities", "duties", "skill",
                    "skills", "competency", "competencies", "benefit", "benefits", "perks", "company",
                    "companies", "employer", "employers", "worked", "work", "employment", "tenure",
                    "professional", "candidate", "candidates", "bde", "business", "development"}
    matches = {}
    for term in _focus_terms(query) - intent_terms:
        if len(term) < 3:
            continue
        document_ids = {chunk.document_id for chunk, _ in rows if term in set(terms(chunk.text))}
        if len(document_ids) == 1:
            matches[term] = document_ids
    return set().union(*matches.values()) if matches else set()


def select_context(candidates, rows, settings, tokenizer, query=None, semantic_intent=None):
    vectors = {chunk.chunk_id: vector for chunk, vector in rows}
    all_candidates = list(candidates)
    remaining, selected, used = list(candidates), [], 0
    aggregate = bool(query and _is_aggregate_query(query))
    document_coverage = bool(query and requires_document_coverage(query, semantic_intent))
    role_differentiation = bool(query and requests_role_differentiation(query))
    summary_preferred = bool(query and (prefers_professional_summary(query)
                                        or role_differentiation))
    preferred_sections = preferred_section_kinds(query or "")
    named_documents = explicitly_matched_document_ids(query, rows) if query else set()
    if query:
        query_terms = _focus_terms(query)
        overlaps = {candidate.chunk.chunk_id: _term_overlap(query_terms, candidate.chunk.text)
                    for candidate in remaining}
        strongest = max(overlaps.values(), default=0)
        # A uniquely specific lexical match is safer context than near-tied
        # dense candidates containing unrelated lists from the same document.
        if strongest >= 2 and not document_coverage and not preferred_sections:
            remaining = [candidate for candidate in remaining
                         if overlaps[candidate.chunk.chunk_id] == strongest]
    # Rank-normalized relevance avoids mixing uncalibrated reranker logits with cosine.
    relevance = {c.chunk.chunk_id: 1 - i / max(len(candidates), 1)
                 for i, c in enumerate(candidates)}
    while remaining and len(selected) < settings.context_k:
        unseen_documents = [candidate for candidate in remaining
                             if candidate.chunk.document_id not in {item.chunk.document_id for item in selected}]
        pool = unseen_documents if document_coverage and unseen_documents else remaining
        summary_chunks = [candidate for candidate in pool if candidate.chunk.section_kind == "professional_summary"]
        preferred_chunks = [candidate for candidate in pool
                            if candidate.chunk.section_kind in preferred_sections
                            and (not named_documents or candidate.chunk.document_id in named_documents)]
        if role_differentiation:
            summary_documents = {candidate.chunk.document_id for candidate in selected
                                 if candidate.chunk.section_kind == "professional_summary"}
            role_summaries = [candidate for candidate in remaining
                              if candidate.chunk.document_id not in summary_documents
                              and candidate.chunk.section_kind == "professional_summary"]
            supporting_documents = {candidate.chunk.document_id for candidate in selected
                                    if candidate.chunk.section_kind in {"skills", "professional_experience", "responsibilities"}}
            documents_needing_support = summary_documents - supporting_documents
            support_chunks = [candidate for candidate in remaining
                              if candidate.chunk.document_id in documents_needing_support
                              and candidate.chunk.section_kind == "skills"]
            if not support_chunks:
                support_chunks = [candidate for candidate in remaining
                                  if candidate.chunk.document_id in documents_needing_support
                                  and candidate.chunk.section_kind in {"professional_experience", "responsibilities"}]
            if role_summaries:
                best = role_summaries[0]
            elif support_chunks:
                best = support_chunks[0]
            elif preferred_chunks:
                best = preferred_chunks[0]
            elif summary_chunks:
                best = summary_chunks[0]
            elif document_coverage and unseen_documents:
                best = unseen_documents[0]
            else:
                best = remaining[0]
        elif preferred_chunks:
            best = preferred_chunks[0]
        elif summary_preferred and summary_chunks:
            best = summary_chunks[0]
        elif document_coverage and unseen_documents:
            # A candidate summary is incomplete if top-ranked chunks all come from one resume.
            best = unseen_documents[0]
        elif settings.mmr and selected:
            def utility(candidate):
                vector = vectors[candidate.chunk.chunk_id]
                redundancy = max(dot(vector, vectors[c.chunk.chunk_id]) for c in selected)
                return settings.mmr_lambda * relevance[candidate.chunk.chunk_id] - (1 - settings.mmr_lambda) * redundancy
            best = max(remaining, key=utility)
        else:
            best = remaining[0]
        remaining.remove(best)
        size = len(tokenizer.encode(best.chunk.text, add_special_tokens=False))
        if used + size <= settings.context_tokens:
            selected.append(best)
            used += size
            if aggregate and not document_coverage and best.chunk.page is not None:
                siblings = [candidate for candidate in all_candidates
                            if candidate.chunk.document_id == best.chunk.document_id
                            and candidate.chunk.page == best.chunk.page
                            and candidate not in selected]
                remaining = siblings + [candidate for candidate in remaining if candidate not in siblings]
    return selected
