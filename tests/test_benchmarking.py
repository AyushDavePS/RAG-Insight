from rag_insight.benchmarking import benchmark_embedding_batches
from rag_insight.config import Settings


def test_batch_benchmark_records_each_requested_size(monkeypatch, tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("evidence", encoding="utf-8")

    class Embedder:
        tokenizer = type("Tokenizer", (), {"encode": lambda _, text, **kwargs: list(text)})()

        def __init__(self, model, url, batch_size=32):
            self.batch_size = batch_size

        def encode_batch(self, texts):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("rag_insight.benchmarking.OllamaEmbedder", Embedder)
    result = benchmark_embedding_batches(Settings(chunk_tokens=100), [source], batch_sizes=(1, 2))
    assert result["embedding_model"] == "nomic-embed-text:latest"
    assert [record["batch_size"] for record in result["records"]] == [1, 2]
    assert all(record["vectors"] == 1 and record["dimension"] == 2 for record in result["records"])
