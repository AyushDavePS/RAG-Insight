import pytest

from rag_insight.generation import generate, grade
from rag_insight.models import Candidate, Chunk


class FakeLLM:
    def __init__(self, result):
        self.result = result

    def json(self, *args):
        return self.result


def test_unknown_citation_is_rejected():
    llm = FakeLLM({"claims": [{"text": "Claim", "source_ids": ["invented"]}]})
    context = [Candidate(Chunk("doc", "v1", "a.md", "Evidence", chunk_id="real"), 1)]
    with pytest.raises(ValueError, match="unknown source"):
        generate(llm, "Question", context)


def test_string_false_is_not_treated_as_true():
    context = [Candidate(Chunk("doc", "v1", "a.md", "Evidence", chunk_id="real"), 1)]
    with pytest.raises(ValueError, match="Malformed"):
        grade(FakeLLM({"sufficient": "false", "reason": "No evidence"}), "Question", context)
