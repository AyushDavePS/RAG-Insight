import json
from time import perf_counter

from .chunking import chunk_sections
from .generation import ABSTENTION, generate, grade, rewrite
from .ingestion import parse_document
from .models import Answer
from .retrieval import bm25_search, dense_search, fuse, select_context


class Pipeline:
    def __init__(self, settings, store, embedder, reranker, llm):
        self.settings, self.store = settings, store
        self.embedder, self.reranker, self.llm = embedder, reranker, llm

    def signature(self):
        s = self.settings
        return json.dumps({"embedding": s.embedding_model, "strategy": s.chunk_strategy,
                           "backend": s.embedding_backend,
                           "size": s.chunk_tokens, "overlap": s.overlap_tokens,
                           "schema": 1}, sort_keys=True)

    def ingest(self, path):
        sections = parse_document(path)
        chunks = chunk_sections(sections, self.settings, self.embedder.tokenizer)
        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        self.store.replace_document(sections[0].document_id, chunks, vectors, self.signature())
        return len(chunks)

    def retrieve(self, query):
        s = self.settings
        if self.store.signature() not in {None, self.signature()}:
            raise ValueError("Settings do not match the index configuration")
        rows = self.store.all()
        start = perf_counter()
        dense = dense_search(self.embedder.encode([query])[0], rows, s.candidate_k) if rows and s.retrieval_mode != "bm25" else []
        lexical = bm25_search(query, rows, s.candidate_k) if s.retrieval_mode != "dense" else []
        merged = fuse([dense, lexical], s.rrf_k) if s.retrieval_mode == "hybrid" else dense or lexical
        candidates = merged[:s.candidate_k]
        if s.rerank:
            candidates = self.reranker.rank(query, candidates)
        context = select_context(candidates, rows, s, self.embedder.tokenizer)
        trace = {"query": query, "dense": len(dense), "bm25": len(lexical),
                 "merged": len(merged), "reranked": len(candidates) if s.rerank else 0,
                 "selected": len(context), "retrieval_seconds": perf_counter() - start,
                 "ranking": [{"chunk_id": c.chunk.chunk_id, "score": c.score,
                              "filename": c.chunk.filename} for c in candidates],
                 "context_ids": [c.chunk.chunk_id for c in context]}
        return context, trace

    def ask(self, question):
        question = question.strip()
        if not question:
            raise ValueError("Enter a question")
        start, trace, query = perf_counter(), [], question
        for attempt in range(2 if self.settings.corrective else 1):
            context, step = self.retrieve(query)
            assessment = grade(self.llm, question, context)
            step.update(attempt=attempt, sufficient=assessment.sufficient, reason=assessment.reason)
            trace.append(step)
            if assessment.sufficient:
                text, sources = generate(self.llm, question, context)
                step["total_seconds"] = perf_counter() - start
                if text:
                    return Answer(text, sources, True, assessment.reason, trace)
                return Answer(ABSTENTION, [], False, "Generation produced no supported claims", trace)
            if attempt == 0 and self.settings.corrective:
                query = rewrite(self.llm, question, assessment.reason)
        trace[-1]["total_seconds"] = perf_counter() - start
        return Answer(ABSTENTION, [], False, assessment.reason, trace)
