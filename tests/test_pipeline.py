from rag_insight.config import Settings
from rag_insight.generation import ABSTENTION
from rag_insight.models import Candidate, Chunk
from rag_insight.pipeline import Pipeline


def test_retry_is_bounded_and_original_question_is_graded():
    class FakeLLM:
        def __init__(self):
            self.graded = []

        def json(self, system, payload):
            if "original_question" in payload:
                return {"query": "rewritten search"}
            self.graded.append(payload["question"])
            return {"sufficient": False, "reason": "Missing explanation"}

    llm = FakeLLM()
    pipeline = Pipeline(Settings(), None, None, None, llm)
    queries = []

    def retrieve(query):
        queries.append(query)
        return [Candidate(Chunk("doc", "v1", "a.md", "Redis", chunk_id="a"), 1)], {}

    pipeline.retrieve = retrieve
    answer = pipeline.ask("Why Redis?")
    assert queries == ["Why Redis?", "rewritten search"]
    assert llm.graded == ["Why Redis?", "Why Redis?"]
    assert answer.text == ABSTENTION
    assert not answer.sufficient
