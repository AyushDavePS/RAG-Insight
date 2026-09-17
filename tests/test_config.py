from rag_insight.config import Settings


def test_default_context_and_reranking_configuration():
    settings = Settings()
    assert settings.rerank is True
    assert settings.context_k == 7
    assert settings.context_tokens == 2000
