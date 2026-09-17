"""Small, reproducible local embedding-throughput measurements."""
from time import perf_counter

from .chunking import chunk_sections
from .embeddings import OllamaEmbedder
from .ingestion import parse_document


def benchmark_embedding_batches(settings, paths, batch_sizes=(16, 32, 64, 128)):
    """Measure real Ollama batch calls on a fixed local document set."""
    if not batch_sizes or any(type(size) is not int or size < 1 for size in batch_sizes):
        raise ValueError("Batch sizes must be positive integers")
    tokenizer = OllamaEmbedder(settings.embedding_model, settings.ollama_url).tokenizer
    chunks = [
        chunk
        for path in paths
        for chunk in chunk_sections(parse_document(path), settings, tokenizer)
    ]
    texts = [chunk.text for chunk in chunks]
    records = []
    for batch_size in batch_sizes:
        embedder = OllamaEmbedder(settings.embedding_model, settings.ollama_url, batch_size)
        started = perf_counter()
        vectors = embedder.encode_batch(texts)
        seconds = perf_counter() - started
        records.append({
            "batch_size": batch_size,
            "chunks": len(texts),
            "vectors": len(vectors),
            "dimension": len(vectors[0]) if vectors else None,
            "seconds": seconds,
        })
    return {"embedding_model": settings.embedding_model, "records": records}
