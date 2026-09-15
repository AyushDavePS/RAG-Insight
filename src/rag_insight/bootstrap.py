from pathlib import Path

from .embeddings import OllamaEmbedder, SentenceTransformerEmbedder
from .llm import LLM
from .pipeline import Pipeline
from .reranking import Reranker
from .storage import Store


def build(settings, index: Path):
    from dotenv import load_dotenv

    load_dotenv()
    if settings.embedding_backend == "ollama":
        embedder = OllamaEmbedder(settings.embedding_model, settings.ollama_url)
    else:
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
    return Pipeline(settings, Store(index), embedder,
                    Reranker(settings.reranker_model) if settings.rerank else None, LLM())
