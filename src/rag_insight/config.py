from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    embedding_backend: str = "ollama"
    chunk_strategy: str = "structure"
    chunk_tokens: int = 500
    overlap_tokens: int = 30
    embedding_model: str = "nomic-embed-text:latest"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_mode: str = "hybrid"
    candidate_k: int = 20
    context_k: int = 5
    context_tokens: int = 1800
    rrf_k: int = 60
    rerank: bool = True
    mmr: bool = False
    mmr_lambda: float = 0.7
    corrective: bool = True
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:latest"
    llm_url: str = "http://localhost:11434/api/chat"

    def __post_init__(self):
        if self.embedding_backend not in {"ollama", "sentence_transformers"}:
            raise ValueError("Unknown embedding backend")
        if self.chunk_strategy not in {"recursive", "structure"}:
            raise ValueError("Unknown chunk strategy")
        if self.retrieval_mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        if not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError("Overlap must be smaller than chunk size")
        if min(self.candidate_k, self.context_k, self.context_tokens, self.rrf_k) < 1:
            raise ValueError("Retrieval limits must be positive")
        if not 0 <= self.mmr_lambda <= 1:
            raise ValueError("MMR lambda must be between zero and one")
