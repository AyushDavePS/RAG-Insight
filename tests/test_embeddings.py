import json

import pytest

from rag_insight.embeddings import OllamaEmbedder


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(())

    def read(self):
        return json.dumps(self.payload).encode()


def test_ollama_adapter_normalizes_and_preserves_batch_order(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        batch = json.loads(request.data.decode())["input"]
        requests.append(batch)
        return Response({"embeddings": [[3 if text == "first" else 4, 0.0] for text in batch]})

    monkeypatch.setattr("rag_insight.embeddings.urlopen", fake_urlopen)
    vectors = OllamaEmbedder("nomic-embed-text").encode(["first", "second"])
    assert requests == [["first", "second"]]
    assert vectors[0] == [1.0, 0.0]
    assert vectors[1] == [1.0, 0.0]


def test_ollama_adapter_rejects_zero_or_inconsistent_vectors(monkeypatch):
    responses = iter([{"embeddings": [[0.0, 0.0]]}])
    monkeypatch.setattr("rag_insight.embeddings.urlopen", lambda request, timeout: Response(next(responses)))
    with pytest.raises(RuntimeError, match="zero vector"):
        OllamaEmbedder("nomic-embed-text").encode(["zero"])

    responses = iter([{"embeddings": [[1.0, 0.0], [1.0, 0.0, 0.0]]}])
    monkeypatch.setattr("rag_insight.embeddings.urlopen", lambda request, timeout: Response(next(responses)))
    with pytest.raises(RuntimeError, match="inconsistent dimensions"):
        OllamaEmbedder("nomic-embed-text").encode(["one", "two"])


def test_ollama_adapter_respects_configured_batch_size(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        batch = json.loads(request.data.decode())["input"]
        requests.append(batch)
        return Response({"embeddings": [[1.0, 0.0] for _ in batch]})

    monkeypatch.setattr("rag_insight.embeddings.urlopen", fake_urlopen)
    vectors = OllamaEmbedder("nomic-embed-text", batch_size=2).encode_batch(["a", "b", "c"])
    assert requests == [["a", "b"], ["c"]]
    assert vectors == [[1.0, 0.0]] * 3
