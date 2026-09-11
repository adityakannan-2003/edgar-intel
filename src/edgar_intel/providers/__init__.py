"""Provider factories. One place that decides what the rest of the code talks to."""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from .base import Completion, EmbeddingProvider, LLMProvider, Reranker

__all__ = [
    "Completion",
    "EmbeddingProvider",
    "LLMProvider",
    "Reranker",
    "get_llm",
    "get_embedder",
    "get_reranker",
    "reset_provider_cache",
]


@lru_cache(maxsize=1)
def get_llm() -> LLMProvider:
    s = get_settings()
    if s.llm_provider == "fake":
        from .fake import FakeLLM

        return FakeLLM()
    from .openai_compat import OpenAICompatLLM

    return OpenAICompatLLM()


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    s = get_settings()
    if s.embed_provider == "fake":
        from .fake import FakeEmbedder

        return FakeEmbedder(dim=s.embed_dim)
    if s.embed_provider == "api":
        from .openai_compat import OpenAICompatEmbedder

        return OpenAICompatEmbedder()
    from .local import LocalEmbedder

    return LocalEmbedder()


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    s = get_settings()
    if s.embed_provider == "fake":
        from .fake import FakeReranker

        return FakeReranker()
    from .local import LocalReranker

    return LocalReranker()


def reset_provider_cache() -> None:
    get_llm.cache_clear()
    get_embedder.cache_clear()
    get_reranker.cache_clear()
