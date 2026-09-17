import pytest

from rag_insight.agent import RetrieveTool
from rag_insight.config import Settings
from rag_insight.generation import ABSTENTION, EmploymentFact, ExperienceFact
from rag_insight.models import Candidate, Chunk
from rag_insight.pipeline import Pipeline
from rag_insight.sources import SourceDocument
from rag_insight.storage import Store


def test_retry_is_bounded_and_original_question_is_graded():
    class FakeLLM:
        def __init__(self):
            self.graded = []

        def json(self, system, payload, **kwargs):
            if "original_question" in payload:
                return {"query": "rewritten search"}
            self.graded.append(payload["question"])
            return {"sufficient": False, "reason": "Missing explanation"}

    llm = FakeLLM()
    pipeline = Pipeline(Settings(), None, None, None, llm)
    queries = []

    def retrieve(query, **kwargs):
        queries.append(query)
        return [Candidate(Chunk("doc", "v1", "a.md", "Redis", chunk_id="a"), 1)], {}

    pipeline.retrieve = retrieve
    answer = pipeline.ask("Why Redis?")
    assert queries == ["Why Redis?", "rewritten search"]
    assert llm.graded == ["Why Redis?", "Why Redis?"]
    assert answer.text == ABSTENTION
    assert not answer.sufficient


def test_reingesting_unchanged_content_reuses_cached_embeddings(tmp_path):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text)

    class Embedder:
        tokenizer = Tokenizer()

        def __init__(self):
            self.calls = []

        def encode_batch(self, texts):
            self.calls.append(texts)
            return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]

    source = tmp_path / "source.txt"
    source.write_text("same content", encoding="utf-8")
    embedder = Embedder()
    settings = Settings(chunk_strategy="recursive", chunk_tokens=100)
    with Store(tmp_path / "index.sqlite") as store:
        pipeline = Pipeline(settings, store, embedder, None, None)
        pipeline.ingest(source)
        pipeline.ingest(source)
        assert embedder.calls == [["same content"]]
        chunk, vector = store.all()[0]
        assert chunk.content_hash
        assert chunk.embedding_model == settings.embedding_model
        assert chunk.embedding_dimension == len(vector)


def test_failed_embedding_batch_preserves_previous_document(tmp_path):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text)

    class Embedder:
        tokenizer = Tokenizer()

        def __init__(self):
            self.calls = 0

        def encode_batch(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("Ollama unavailable")
            return [[1.0, 0.0] for _ in texts]

    source = tmp_path / "source.txt"
    source.write_text("first", encoding="utf-8")
    with Store(tmp_path / "index.sqlite") as store:
        pipeline = Pipeline(Settings(chunk_strategy="recursive", chunk_tokens=100), store, Embedder(), None, None)
        pipeline.ingest(source)
        source.write_text("changed", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Ollama unavailable"):
            pipeline.ingest(source)
        assert [chunk.text for chunk, _ in store.all()] == ["first"]


def test_retrieve_tool_rejects_mutating_or_malformed_calls():
    tool = RetrieveTool(lambda query: ([], {"query": query}), max_top_k=2)
    with pytest.raises(ValueError, match="Malformed"):
        tool.execute({"query": "question", "path": "document.txt"})
    with pytest.raises(ValueError, match="top_k"):
        tool.execute({"query": "question", "top_k": 3})


def test_agent_initial_call_uses_history_and_corrective_retry_is_bounded():
    class FakeLLM:
        def __init__(self):
            self.calls = []

        def json(self, system, payload, **kwargs):
            self.calls.append(payload)
            if "conversation" in payload:
                return {"query": "Redis decision", "top_k": 1}
            if "original_question" in payload:
                return {"query": "Redis rationale"}
            return {"sufficient": False, "reason": "Need more evidence"}

    llm = FakeLLM()
    pipeline = Pipeline(Settings(agentic=True, corrective=True), None, None, None, llm)
    queries = []

    def retrieve(query, **kwargs):
        queries.append(query)
        return [Candidate(Chunk("doc", "v1", "a.md", "Redis", chunk_id="a"), 1)], {"query": query}

    pipeline.retrieve = retrieve
    answer = pipeline.ask("Why Redis?", history=[{"role": "user", "content": "We discussed caching."}])
    assert queries == ["Redis decision", "Redis rationale"]
    assert llm.calls[0]["conversation"][0]["content"] == "We discussed caching."
    assert answer.trace[0]["agent"]["initial_tool_calls"] == 1
    assert len(answer.trace) == 2


def test_agent_excludes_previously_cited_documents_for_remaining_follow_up():
    class FakeLLM:
        def json(self, system, payload, **kwargs):
            if "conversation" in payload:
                return {"query": "BDE candidate summary", "top_k": 3}
            if "retrieval_problem" in payload:
                return {"query": "remaining BDE candidates"}
            return {"sufficient": False, "reason": "No evidence"}

    pipeline = Pipeline(Settings(agentic=True, corrective=False), None, None, None, FakeLLM())
    calls = []

    def retrieve(query, exclude_filenames=None, **kwargs):
        calls.append((query, exclude_filenames))
        return [Candidate(Chunk("rohan", "v1", "rohan.pdf", "Rohan", chunk_id="rohan"), 1)], {"query": query}

    pipeline.retrieve = retrieve
    pipeline.ask("other remaining candidates summary", history=[
        {"role": "assistant", "content": "Isha summary [1]", "source_filenames": ["isha.pdf"]},
    ])
    assert calls == [("BDE candidate summary", ["isha.pdf"])]


def test_agent_expands_document_wide_summary_to_context_limit():
    class FakeLLM:
        def json(self, system, payload, **kwargs):
            if "conversation" in payload:
                return {"query": "professional summaries", "top_k": 1}
            if "evidence" in payload:
                return {"claims": [{"text": "summary", "source_ids": [payload["evidence"][0]["chunk_id"]]}]}
            return {"sufficient": False, "reason": "No evidence"}

    pipeline = Pipeline(Settings(agentic=True, corrective=False, context_k=5), None, None, None, FakeLLM())
    calls = []

    def retrieve(query, **kwargs):
        calls.append(query)
        context = [Candidate(Chunk(str(index), "v1", f"{index}.pdf", "summary", chunk_id=str(index)), 1)
                   for index in range(3)]
        return context, {"query": query, "document_coverage_target": 3}

    pipeline.retrieve = retrieve
    answer = pipeline.ask("List professional summaries from given documents")
    assert calls == ["professional summaries"]
    assert len(answer.trace[0]["tool"]["evidence"]) == 3


def test_agent_semantic_intent_broadens_indirect_comparison_coverage():
    class FakeLLM:
        def json(self, system, payload, **kwargs):
            if "conversation" in payload:
                return {"query": "longest career", "top_k": 1,
                        "collection_wide": True, "comparative": True}
            if "candidates" in payload:
                return {"selected_document_id": "0", "criteria": ["documented tenure"],
                        "rationale": "Candidate 0 has the longest tenure.", "supporting_source_ids": ["0"]}
            return {"sufficient": False, "reason": "test only"}

    pipeline = Pipeline(Settings(agentic=True, corrective=False, context_k=5), None, None, None, FakeLLM())
    calls = []

    def retrieve(query, **kwargs):
        calls.append((query, kwargs["query_intent"]))
        context = [Candidate(Chunk(str(index), "v1", f"{index}.pdf", "experience", chunk_id=str(index)), 1)
                   for index in range(5)]
        return context, {"query": query, "document_coverage_target": 5}

    pipeline.retrieve = retrieve
    answer = pipeline.ask("Who has spent the longest time in the field?")
    assert calls == [("longest career", {
        "collection_wide": True, "comparative": True, "aggregation": False,
    })]
    assert answer.trace[0]["agent"]["semantic_intent"]["comparative"]
    assert len(answer.trace[0]["tool"]["evidence"]) == 5


def test_agent_marks_collection_aggregation_for_full_coverage_even_when_model_omits_it():
    class FakeLLM:
        def json(self, system, payload, **kwargs):
            if "conversation" in payload:
                return {"query": "candidate employer count", "top_k": 1,
                        "collection_wide": False, "comparative": False, "aggregation": False}
            return {"sufficient": False, "reason": "test only"}

    pipeline = Pipeline(Settings(agentic=True, corrective=False, context_k=5), None, None, None, FakeLLM())
    calls = []

    def retrieve(query, **kwargs):
        calls.append((query, kwargs["query_intent"]))
        return [], {"query": query, "document_coverage_target": 0}

    pipeline.retrieve = retrieve
    answer = pipeline.ask("How many candidate profiles have more than one employer?")
    assert calls == [("candidate employer count", {
        "collection_wide": False, "comparative": False, "aggregation": True,
    })]
    assert answer.trace[0]["agent"]["plan"] == "structured_aggregation"


def test_numeric_comparison_uses_map_reduce_evidence_not_the_llm_context_budget():
    pipeline = Pipeline(Settings(agentic=False, corrective=False, context_k=1), None, None, None, None)
    facts = [
        Candidate(Chunk("aarav", "v1", "aarav.pdf", "Aarav\n2.8 years of experience", chunk_id="aarav"), 2.8),
        Candidate(Chunk("isha", "v1", "isha.pdf", "Isha\n4 years of experience", chunk_id="isha"), 4.0),
        Candidate(Chunk("neha", "v1", "neha.pdf", "Neha\n3.5 years of experience", chunk_id="neha"), 3.5),
    ]

    def retrieve(query, **kwargs):
        ordinary_context = [Candidate(Chunk("aarav", "v1", "aarav.pdf", "Long chunk", chunk_id="ordinary"), 1)]
        return ordinary_context, {"query": query, "document_coverage_target": 3,
                                  "_comparison_candidates": facts}

    pipeline.retrieve = retrieve
    answer = pipeline.ask("Which candidate has the highest years of experience?")
    assert answer.text.startswith("Isha has the highest stated experience: 4 years.")
    assert [source.chunk_id for source in answer.sources] == ["isha", "neha", "aarav"]


def test_count_and_highest_experience_compose_from_complete_structured_facts():
    pipeline = Pipeline(Settings(agentic=False, corrective=False, context_k=1), None, None, None, None)
    maya_profile = Chunk("maya", "v1", "maya.pdf", "Maya Joshi", chunk_id="maya-profile")
    maya_work = Chunk("maya", "v1", "maya.pdf", "Engineer — BuildLab | Jan 2021 - Jun 2025",
                      chunk_id="maya-work")
    sana = Chunk("sana", "v1", "sana.pdf", "Sana Khan\n3 years of experience", chunk_id="sana-profile")
    facts = [
        ExperienceFact("maya", "Maya Joshi", 4.5, [maya_profile, maya_work], True),
        ExperienceFact("sana", "Sana Khan", 3.0, [sana], False),
    ]

    def retrieve(query, **kwargs):
        ordinary_context = [Candidate(sana, 1)]
        return ordinary_context, {"query": query, "document_coverage_target": 2,
                                  "_experience_facts": facts}

    pipeline.retrieve = retrieve
    answer = pipeline.ask("How many Fullstack AIML candidates are there and who has the highest experience?")
    assert answer.sufficient
    assert answer.text.startswith("There are 2 candidates in scope. Maya Joshi has the highest documented employment duration")
    assert [source.chunk_id for source in answer.sources] == ["maya-profile", "maya-work", "sana-profile"]


def test_employer_count_uses_full_document_map_reduce_not_context_budget():
    pipeline = Pipeline(Settings(agentic=False, corrective=False, context_k=1), None, None, None, None)
    profile = Chunk("aarav", "v1", "aarav.pdf", "Aarav Mehta", chunk_id="aarav-profile")
    work = Chunk("aarav", "v1", "aarav.pdf", "BDE — CloudNexa | A\nAssociate — LeadOrbit | A",
                 chunk_id="aarav-work")
    isha = Chunk("isha", "v1", "isha.pdf", "Isha Shah\nBDE — VertexWave | M", chunk_id="isha-work")
    facts = [
        EmploymentFact("aarav", "Aarav Mehta", ["CloudNexa", "LeadOrbit"], [profile, work]),
        EmploymentFact("isha", "Isha Shah", ["VertexWave"], [isha]),
    ]

    def retrieve(query, **kwargs):
        ordinary_context = [Candidate(Chunk("aarav", "v1", "aarav.pdf", "Long chunk", chunk_id="ordinary"), 1)]
        return ordinary_context, {"query": query, "document_coverage_target": 2,
                                  "_employment_facts": facts}

    pipeline.retrieve = retrieve
    answer = pipeline.ask("List candidates who have worked in more than 1 company")
    assert answer.sufficient
    assert answer.text == "Aarav Mehta: CloudNexa, LeadOrbit (2 companies) [1] [2]"
    assert [source.chunk_id for source in answer.sources] == ["aarav-profile", "aarav-work"]


def test_retrieve_refreshes_stored_role_metadata_without_reindexing():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()

    class StoreWithOldMetadata:
        def signature(self):
            return None

        def all(self):
            return [
                (Chunk("maya", "v1", "maya.pdf", "Maya Joshi\nFull Stack Developer — AIML",
                       chunk_id="maya"), [1.0]),
                (Chunk("sana", "v1", "sana.pdf", "Sana Khan\nAIML Engineer", chunk_id="sana"), [1.0]),
                (Chunk("rohan", "v1", "rohan.pdf", "Rohan Patel\nBusiness Development Executive",
                       chunk_id="rohan"), [1.0]),
            ]

    embedder = type("Embedder", (), {"tokenizer": Tokenizer()})()
    pipeline = Pipeline(Settings(retrieval_mode="bm25", candidate_k=5, context_k=5, rerank=False),
                        StoreWithOldMetadata(), embedder, None, None)
    context, trace = pipeline.retrieve("List out the Fullstack AIML and just AIML candidates.")
    assert trace["requested_profile_roles"] == ["aiml", "fullstack_aiml"]
    assert trace["profile_filtered_documents"] == 2
    assert {candidate.chunk.filename for candidate in context} == {"maya.pdf", "sana.pdf"}


def test_corrective_retry_preserves_original_document_wide_selection_intent():
    class FakeLLM:
        def json(self, system, payload, **kwargs):
            if "conversation" in payload:
                return {"query": "candidate summary", "top_k": 5}
            if "retrieval_problem" in payload:
                return {"query": "rewritten query"}
            return {"sufficient": False, "reason": "Need another search"}

    pipeline = Pipeline(Settings(agentic=True, corrective=True), None, None, None, FakeLLM())
    calls = []

    def retrieve(query, **kwargs):
        calls.append((query, kwargs["intent_query"]))
        context = [Candidate(Chunk(str(index), "v1", f"{index}.pdf", "summary", chunk_id=str(index)), 1)
                   for index in range(3)]
        return context, {"query": query, "document_coverage_target": 0}

    pipeline.retrieve = retrieve
    pipeline.ask("List professional summaries from given documents")
    assert calls == [
        ("candidate summary", "List professional summaries from given documents"),
        ("rewritten query", "List professional summaries from given documents"),
    ]


def test_ingest_source_preserves_adapter_provenance(tmp_path):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text)

    class Embedder:
        tokenizer = Tokenizer()

        def encode_batch(self, texts):
            return [[1.0, 0.0] for _ in texts]

    source = SourceDocument("file:///fixture.html", "Fixture", "Adapter evidence.", "2026-09-16T00:00:00+00:00", "hash")
    with Store(tmp_path / "index.sqlite") as store:
        pipeline = Pipeline(Settings(chunk_strategy="recursive", chunk_tokens=100), store, Embedder(), None, None)
        pipeline.ingest_source(source)
        chunk = store.all()[0][0]
        assert chunk.source_uri == source.uri
        assert chunk.retrieved_at == source.retrieved_at
