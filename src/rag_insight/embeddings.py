import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .inspection import CharacterBudgetTokenizer


def _normalize(vector):
    if not vector or any(isinstance(value, bool) or not isinstance(value, (int, float))
                         or not math.isfinite(value) for value in vector):
        raise RuntimeError("Embedding model returned invalid vector values")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise RuntimeError("Embedding model returned a zero vector")
    return [float(value) / norm for value in vector]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.encode_batch(texts)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class OllamaEmbedder:
    """Embedding adapter for Ollama's local /api/embed endpoint."""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", batch_size: int = 32):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        if batch_size < 1:
            raise ValueError("Embedding batch size must be positive")
        self.batch_size = batch_size
        # Ollama does not expose a portable Python tokenizer; this is a
        # conservative bound for chunk construction.
        self.tokenizer = CharacterBudgetTokenizer()

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.encode_batch(texts)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            payload = json.dumps({"model": self.model_name, "input": batch}).encode()
            request = Request(f"{self.base_url}/api/embed", data=payload,
                              headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urlopen(request, timeout=120) as response:
                    result = json.load(response)
            except (HTTPError, URLError, TimeoutError) as error:
                raise RuntimeError(f"Ollama embedding request failed at {self.base_url}: {error}") from error
            vectors = result.get("embeddings")
            if vectors is None and len(batch) == 1:
                vectors = [result.get("embedding")]
            if not isinstance(vectors, list) or len(vectors) != len(batch):
                raise RuntimeError("Ollama returned an invalid embedding response")
            try:
                embeddings.extend(_normalize(vector) for vector in vectors)
            except (TypeError, ValueError) as error:
                raise RuntimeError("Ollama returned an invalid embedding response") from error
        dimensions = {len(vector) for vector in embeddings}
        if len(dimensions) != 1:
            raise RuntimeError("Ollama returned embeddings with inconsistent dimensions")
        return embeddings
