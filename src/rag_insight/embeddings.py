import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .inspection import CharacterBudgetTokenizer


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class OllamaEmbedder:
    """Embedding adapter for Ollama's local /api/embed endpoint."""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        # Ollama does not expose a portable Python tokenizer; this is a
        # conservative bound for chunk construction.
        self.tokenizer = CharacterBudgetTokenizer()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = []
        for text in texts:
            payload = json.dumps({"model": self.model_name, "input": text}).encode()
            request = Request(f"{self.base_url}/api/embed", data=payload,
                              headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urlopen(request, timeout=120) as response:
                    result = json.load(response)
            except (HTTPError, URLError, TimeoutError) as error:
                raise RuntimeError(f"Ollama embedding request failed at {self.base_url}: {error}") from error
            vector = result.get("embeddings", [None])[0]
            if vector is None:
                vector = result.get("embedding")
            if not isinstance(vector, list):
                raise RuntimeError("Ollama returned an invalid embedding response")
            embeddings.append(vector)
        return embeddings
