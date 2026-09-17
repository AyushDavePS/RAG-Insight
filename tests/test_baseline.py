import json

from rag_insight.baseline import run_baseline, write_baseline
from rag_insight.config import Settings
from rag_insight.models import Candidate, Chunk


class FakePipeline:
    settings = Settings(embedding_backend="ollama", embedding_model="test-model", rerank=False)

    def retrieve(self, question):
        chunk = Chunk("doc", "version", "test.md", question, line_start=1, line_end=1, chunk_id="id")
        return [Candidate(chunk, 1.0)], {"embedding_dimension": 2, "context_ids": ["id"]}


def test_baseline_trace_is_serializable_and_records_model_metadata(tmp_path):
    result = run_baseline(FakePipeline(), queries=["Question"], expected_evidence={"Question": []})
    assert result["stage"] == "hybrid_dense"
    assert result["embedding"]["dimension"] == 2
    assert result["records"][0]["evidence_labels_selected"] == []
    output = tmp_path / "trace.json"
    write_baseline(result, output)
    assert json.loads(output.read_text(encoding="utf-8"))["records"][0]["question"] == "Question"
