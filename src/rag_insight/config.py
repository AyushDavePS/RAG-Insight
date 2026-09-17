from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    vector_backend: str = "chroma"
    chroma_collection: str = "rag-insight"
    agentic: bool = False
    ocr_enabled: bool = False
    ocr_language: str = "en"
    ocr_render_dpi: int = 150
    ocr_model: str = "mobile"
    embedding_backend: str = "ollama"
    chunk_strategy: str = "hybrid"
    chunk_tokens: int = 500
    overlap_tokens: int = 30
    embedding_model: str = "nomic-embed-text:latest"
    embedding_batch_size: int = 32
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_mode: str = "hybrid"
    candidate_k: int = 20
    context_k: int = 7
    context_tokens: int = 2000
    rrf_k: int = 60
    rerank: bool = True
    mmr: bool = False
    mmr_lambda: float = 0.7
    corrective: bool = True
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:latest"
    llm_url: str = "http://localhost:11434/api/chat"

    def __post_init__(self):
        if self.vector_backend not in {"sqlite", "chroma"}:
            raise ValueError("Unknown vector backend")
        if not self.chroma_collection or len(self.chroma_collection) < 3:
            raise ValueError("Chroma collection name must contain at least three characters")
        if type(self.agentic) is not bool:
            raise ValueError("Agentic mode must be a boolean")
        if type(self.ocr_enabled) is not bool:
            raise ValueError("OCR enabled must be a boolean")
        if not self.ocr_language.strip():
            raise ValueError("OCR language is required")
        if self.ocr_render_dpi < 72:
            raise ValueError("OCR render DPI must be at least 72")
        if self.ocr_model not in {"mobile", "server"}:
            raise ValueError("OCR model must be mobile or server")
        if self.embedding_backend not in {"ollama", "sentence_transformers"}:
            raise ValueError("Unknown embedding backend")
        if self.chunk_strategy not in {"recursive", "structure", "hybrid"}:
            raise ValueError("Unknown chunk strategy")
        if self.retrieval_mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        if not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError("Overlap must be smaller than chunk size")
        if min(self.candidate_k, self.context_k, self.context_tokens, self.rrf_k) < 1:
            raise ValueError("Retrieval limits must be positive")
        if self.embedding_batch_size < 1:
            raise ValueError("Embedding batch size must be positive")
        if not 0 <= self.mmr_lambda <= 1:
            raise ValueError("MMR lambda must be between zero and one")
