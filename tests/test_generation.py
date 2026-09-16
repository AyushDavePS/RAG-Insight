import pytest

from rag_insight.generation import generate, grade
from rag_insight.models import Candidate, Chunk


class FakeLLM:
    def __init__(self, result):
        self.result = result

    def json(self, *args, **kwargs):
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


def test_table_question_transcribes_extracted_rows_without_an_llm_rewrite():
    context = [Candidate(Chunk("doc", "v1", "table.pdf", "ID Name Role\n001 Jane Developer\n002 Sam Designer\nText after table", chunk_id="table"), 1)]
    text, sources = generate(FakeLLM({}), "What values are inside the table?", context)
    assert text == "ID Name Role\n001 Jane Developer\n002 Sam Designer [1]"
    assert [source.chunk_id for source in sources] == ["table"]


def test_version_year_question_transcribes_all_retrieved_pairs():
    first = Candidate(Chunk("doc", "v1", "history.pdf", "PDF 1.0\n1993\nPDF 1.7\n2006", chunk_id="first"), 1)
    second = Candidate(Chunk("doc", "v1", "history.pdf", "PDF 1.7\n2006\nPDF 2.0\n2017", chunk_id="second"), 0.9)
    text, sources = generate(FakeLLM({}), "List all PDF versions and years", [first, second])
    assert text == "PDF 1.0 1993\nPDF 1.7 2006\nPDF 2.0 2017 [1] [2]"
    assert [source.chunk_id for source in sources] == ["first", "second"]


def test_regional_distribution_transcribes_every_vertical_table_row():
    text = "Region\nUsers\nShare (%)\nAvg. Pages/Visit\nTop Browser\nNorth America\n18,400\n29.5\n3.2\nChrome\nEurope\n16,200\n26.0\n3.5\nChrome\nTable 6.2"
    context = [Candidate(Chunk("doc", "v1", "regions.pdf", text, chunk_id="regions"), 1)]
    answer, sources = generate(FakeLLM({}), "List all regional distribution data", context)
    assert answer == "Region | Users | Share (%) | Avg. Pages/Visit | Top Browser\nNorth America | 18,400 | 29.5 | 3.2 | Chrome\nEurope | 16,200 | 26.0 | 3.5 | Chrome [1]"
    assert [source.chunk_id for source in sources] == ["regions"]
