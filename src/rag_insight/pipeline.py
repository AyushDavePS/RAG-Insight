import json
from time import perf_counter

from .agent import RetrievalAgent, RetrieveTool
from .chunking import chunk_sections
from .generation import (
    ABSTENTION,
    employment_company_facts,
    employment_threshold,
    experience_comparison_facts,
    generate,
    grade,
    is_highest_experience_query,
    render_employment_company_answer,
    render_experience_comparison_answer,
    rewrite,
)
from .ingestion import parse_document, parse_source
from .metadata import enrich_chunks
from .models import Answer, Candidate, Chunk
from .ocr import PaddleOcrExtractor
from .registry import build_document_registry
from .retrieval import (
    bm25_search,
    dense_search,
    explicitly_matched_document_ids,
    filter_rows_by_profile,
    fuse,
    is_comparative_query,
    preferred_section_kinds,
    requires_document_coverage,
    select_context,
)


class Pipeline:
    def __init__(self, settings, store, embedder, reranker, llm):
        self.settings, self.store = settings, store
        self.embedder, self.reranker, self.llm = embedder, reranker, llm

    def signature(self):
        s = self.settings
        return json.dumps({"embedding": s.embedding_model, "strategy": s.chunk_strategy,
                           "backend": s.embedding_backend, "vector_backend": s.vector_backend,
                           "size": s.chunk_tokens, "overlap": s.overlap_tokens,
                           "schema": 4}, sort_keys=True)

    def ingest(self, path):
        extractor = PaddleOcrExtractor(self.settings.ocr_language, self.settings.ocr_render_dpi, self.settings.ocr_model) \
            if self.settings.ocr_enabled else None
        sections = parse_document(path, extractor)
        return self._ingest_sections(sections)

    def ingest_source(self, source):
        return self._ingest_sections(parse_source(source))

    def _ingest_sections(self, sections):
        chunks = chunk_sections(sections, self.settings, self.embedder.tokenizer)
        chunks = enrich_chunks(chunks)
        if self.store.signature() not in {None, self.signature()}:
            raise ValueError("Settings do not match the index configuration")
        cached = self.store.cached_vectors(
            [chunk.content_hash for chunk in chunks], self.settings.embedding_model
        )
        missing_hashes = list(dict.fromkeys(
            chunk.content_hash for chunk in chunks if chunk.content_hash not in cached
        ))
        if missing_hashes:
            text_by_hash = {chunk.content_hash: chunk.text for chunk in chunks}
            new_vectors = self.embedder.encode_batch([text_by_hash[value] for value in missing_hashes])
            if len(new_vectors) != len(missing_hashes):
                raise RuntimeError("Embedding adapter returned an unexpected vector count")
            cached.update(zip(missing_hashes, new_vectors, strict=True))
        vectors = [cached[chunk.content_hash] for chunk in chunks]
        dimension = len(vectors[0])
        chunks = [Chunk(**{**chunk.__dict__, "embedding_model": self.settings.embedding_model,
                           "embedding_dimension": dimension}) for chunk in chunks]
        self.store.replace_document(sections[0].document_id, chunks, vectors, self.signature())
        return len(chunks)

    def clear_collection(self):
        self.store.clear()

    def document_registry(self):
        """Return the current evidence-derived document registry for inspection/planning."""
        chunks = enrich_chunks([chunk for chunk, _ in self.store.all()])
        return build_document_registry(chunks)

    def retrieve(self, query, exclude_filenames=None, intent_query=None, query_intent=None):
        s = self.settings
        if self.store.signature() not in {None, self.signature()}:
            raise ValueError("Settings do not match the index configuration")
        excluded = set(exclude_filenames or [])
        stored_rows = [(chunk, vector) for chunk, vector in self.store.all() if chunk.filename not in excluded]
        # Metadata is deterministic and derived solely from stored chunk text.
        # Refresh it here as well as at ingestion so an existing collection
        # benefits from improved role taxonomy without re-embedding documents.
        refreshed_chunks = {chunk.chunk_id: chunk for chunk in enrich_chunks(
            [chunk for chunk, _ in stored_rows]
        )}
        all_rows = [(refreshed_chunks[chunk.chunk_id], vector) for chunk, vector in stored_rows]
        registry = build_document_registry([chunk for chunk, _ in all_rows])
        context_query = intent_query or query
        rows, requested_roles = filter_rows_by_profile(all_rows, context_query)
        coverage_target = (len({chunk.document_id for chunk, _ in rows})
                           if requires_document_coverage(context_query, query_intent) else 0)
        embedding_start = perf_counter()
        query_vector = self.embedder.encode([query])[0] if rows and s.retrieval_mode != "bm25" else None
        embedding_seconds = perf_counter() - embedding_start
        start = perf_counter()
        # Chroma does not expose our nested chunk metadata as filterable fields. Use the
        # transparent exact scorer after an explicit scope filter so filtered documents
        # can never re-enter via vector search.
        dense = (dense_search(query_vector, rows, s.candidate_k) if requested_roles else
                 self.store.dense_search(query_vector, s.candidate_k)) if query_vector is not None else []
        lexical = bm25_search(query, rows, s.candidate_k) if s.retrieval_mode != "dense" else []
        merged = fuse([dense, lexical], s.rrf_k) if s.retrieval_mode == "hybrid" else dense or lexical
        candidates = merged[:s.candidate_k]
        if s.rerank:
            candidates = self.reranker.rank(query, candidates)
        context = select_context(candidates, rows, s, self.embedder.tokenizer, context_query, query_intent)
        # Comparison facts are reduced directly from indexed chunks. They are
        # deliberately kept separate from LLM context, which remains budgeted.
        comparison_facts = experience_comparison_facts(context_query, [Candidate(chunk, 0.0) for chunk, _ in rows])
        comparison_candidates = [Candidate(fact.sources[0], fact.years) for fact in comparison_facts]
        employment_facts = employment_company_facts(context_query, [Candidate(chunk, 0.0) for chunk, _ in rows])
        trace = {"query": query, "dense": len(dense), "bm25": len(lexical),
                 "merged": len(merged), "reranked": len(candidates) if s.rerank else 0,
                 "selected": len(context), "embedding_seconds": embedding_seconds,
                 "retrieval_seconds": perf_counter() - start,
                 "embedding_backend": s.embedding_backend, "embedding_model": s.embedding_model,
                 "embedding_dimension": len(query_vector) if query_vector is not None else None,
                 "excluded_filenames": sorted(excluded),
                 "requested_profile_roles": sorted(requested_roles),
                 "profile_filtered_documents": len({chunk.document_id for chunk, _ in rows}),
                 "document_coverage_target": coverage_target,
                 "selection_intent": context_query,
                 "semantic_intent": query_intent or {},
                 "preferred_section_kinds": sorted(preferred_section_kinds(context_query)),
                 "explicitly_matched_documents": sorted(explicitly_matched_document_ids(context_query, rows)),
                 "document_registry": [record.trace() for record in registry],
                 "comparison_fact_mode": "years_experience" if comparison_candidates else None,
                 "comparison_fact_coverage": len(comparison_candidates),
                 "comparison_fact_source_ids": [item.chunk.chunk_id for item in comparison_candidates],
                 "employment_fact_mode": "company_count" if employment_threshold(context_query) is not None else None,
                 "employment_fact_coverage": len(employment_facts),
                 "employment_fact_documents": [fact.document_id for fact in employment_facts],
                 "ranking": [{"chunk_id": c.chunk.chunk_id, "score": c.score,
                              "filename": c.chunk.filename} for c in candidates],
                 "context_ids": [c.chunk.chunk_id for c in context]}
        if comparison_candidates:
            trace["_comparison_candidates"] = comparison_candidates
        if comparison_facts:
            trace["_experience_facts"] = comparison_facts
        if employment_facts:
            trace["_employment_facts"] = employment_facts
        return context, trace

    def ask(self, question, history=None):
        question = question.strip()
        if not question:
            raise ValueError("Enter a question")
        start, trace, query = perf_counter(), [], question
        semantic_intent = {}

        def retrieve_for_question(query, exclude_filenames=None, query_intent=None):
            return self.retrieve(query, exclude_filenames=exclude_filenames, intent_query=question,
                                 query_intent=query_intent or semantic_intent)

        tool = RetrieveTool(retrieve_for_question, self.settings.context_k)
        for attempt in range(2 if self.settings.corrective else 1):
            if attempt == 0 and self.settings.agentic:
                result = RetrievalAgent(self.llm, tool, self.settings.llm_model).retrieve(question, history)
                semantic_intent = result.trace.get("agent", {}).get("semantic_intent", {})
            else:
                result = tool.execute({"query": query, "top_k": self.settings.context_k}, semantic_intent)
            context, step = result.context, result.trace
            comparison_context = step.pop("_comparison_candidates", [])
            experience_facts = step.pop("_experience_facts", [])
            employment_facts = step.pop("_employment_facts", [])
            if is_highest_experience_query(question):
                covered = len(experience_facts)
                required = step.get("document_coverage_target", 0)
                sufficient = required > 0 and covered >= required
                reason = ("Every in-scope document has comparable experience evidence."
                          if sufficient else f"Experience evidence covers only {covered} of {required} indexed documents.")
                if sufficient:
                    rendered = render_experience_comparison_answer(question, experience_facts)
                    if rendered is not None:
                        text, sources = rendered
                        step.update(attempt=attempt, sufficient=True, reason=reason,
                                    experience_fact_mode="stated_or_dated_duration")
                        trace.append(step)
                        step["total_seconds"] = perf_counter() - start
                        return Answer(text, sources, True, reason, trace)
                elif attempt == 0 and self.settings.corrective:
                    step.update(attempt=attempt, sufficient=False, reason=reason)
                    trace.append(step)
                    query = rewrite(self.llm, question, reason)
                    continue
            company_threshold = employment_threshold(question)
            if company_threshold is not None:
                covered = len(employment_facts)
                required = step.get("document_coverage_target", 0)
                sufficient = required > 0 and covered >= required
                reason = ("Every in-scope document has structured employment evidence."
                          if sufficient else f"Employment evidence covers only {covered} of {required} indexed documents.")
                step.update(attempt=attempt, sufficient=sufficient, reason=reason)
                trace.append(step)
                if sufficient:
                    text, sources = render_employment_company_answer(employment_facts, company_threshold)
                    step["total_seconds"] = perf_counter() - start
                    return Answer(text, sources, True, reason, trace)
                if attempt == 0 and self.settings.corrective:
                    query = rewrite(self.llm, question, reason)
                    continue
                step["total_seconds"] = perf_counter() - start
                return Answer(ABSTENTION, [], False, reason, trace)
            evidence_context = comparison_context or context
            assessment = grade(self.llm, question, evidence_context, step.get("document_coverage_target", 0))
            step.update(attempt=attempt, sufficient=assessment.sufficient, reason=assessment.reason)
            trace.append(step)
            if assessment.sufficient:
                text, sources = generate(self.llm, question, evidence_context, step.get("document_coverage_target", 0),
                                         is_comparative_query(question, semantic_intent))
                step["total_seconds"] = perf_counter() - start
                if text:
                    return Answer(text, sources, True, assessment.reason, trace)
                return Answer(ABSTENTION, [], False, "Generation produced no supported claims", trace)
            if attempt == 0 and self.settings.corrective:
                query = rewrite(self.llm, question, assessment.reason)
        trace[-1]["total_seconds"] = perf_counter() - start
        return Answer(ABSTENTION, [], False, assessment.reason, trace)
