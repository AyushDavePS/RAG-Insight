from pathlib import Path

from .embeddings import OllamaEmbedder, SentenceTransformerEmbedder
from .llm import LLM
from .pipeline import Pipeline
from .reranking import Reranker
from .storage import ChromaStore, SQLiteExactStore


def build(settings, index: Path):
    from dotenv import load_dotenv

    load_dotenv()
    if settings.embedding_backend == "ollama":
        embedder = OllamaEmbedder(
            settings.embedding_model, settings.ollama_url, settings.embedding_batch_size
        )
    else:
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
    if settings.vector_backend == "chroma":
        store = ChromaStore(index.parent / f"{index.stem}-chroma", settings.chroma_collection)
    else:
        store = SQLiteExactStore(index)
    return Pipeline(settings, store, embedder,
                    Reranker(settings.reranker_model) if settings.rerank else None,
                    LLM(settings.llm_model, settings.llm_url))
