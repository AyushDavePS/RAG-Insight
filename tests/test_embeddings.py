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


def test_ollama_adapter_normalizes_and_preserves_scalar_order(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data.decode())["input"])
        value = 3 if requests[-1] == "first" else 4
        return Response({"embeddings": [[value, 0.0]]})

    monkeypatch.setattr("rag_insight.embeddings.urlopen", fake_urlopen)
    vectors = OllamaEmbedder("nomic-embed-text").encode(["first", "second"])
    assert requests == ["first", "second"]
    assert vectors[0] == [1.0, 0.0]
    assert vectors[1] == [1.0, 0.0]


def test_ollama_adapter_rejects_zero_or_inconsistent_vectors(monkeypatch):
    responses = iter([{"embeddings": [[0.0, 0.0]]}])
    monkeypatch.setattr("rag_insight.embeddings.urlopen", lambda request, timeout: Response(next(responses)))
    with pytest.raises(RuntimeError, match="zero vector"):
        OllamaEmbedder("nomic-embed-text").encode(["zero"])

    responses = iter([{"embeddings": [[1.0, 0.0]]}, {"embeddings": [[1.0, 0.0, 0.0]]}])
    monkeypatch.setattr("rag_insight.embeddings.urlopen", lambda request, timeout: Response(next(responses)))
    with pytest.raises(RuntimeError, match="inconsistent dimensions"):
        OllamaEmbedder("nomic-embed-text").encode(["one", "two"])
