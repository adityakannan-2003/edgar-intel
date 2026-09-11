"""Local sentence-transformers embedding and cross-encoder reranking.

Both run on CPU. That is a deliberate constraint: it keeps the whole retrieval
side of this project free to run, which means you can iterate on chunking and
fusion as many times as you like without watching a spend meter. Only
generation and judging cost money.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings


@lru_cache(maxsize=4)
def _load_encoder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "Local embeddings need the 'models' extra: pip install -e '.[models]'"
        ) from exc
    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Local reranking needs the 'models' extra: pip install -e '.[models]'"
        ) from exc
    return CrossEncoder(model_name)


class LocalEmbedder:
    name = "local"

    def __init__(self, model: str | None = None, dim: int | None = None) -> None:
        s = get_settings()
        self.model_name = model or s.embed_model
        self.dim = dim or s.embed_dim
        self.batch = s.embed_batch

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        enc = _load_encoder(self.model_name)
        vecs = enc.encode(
            texts,
            batch_size=self.batch,
            normalize_embeddings=True,  # cosine distance in pgvector assumes this
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vecs]


class LocalReranker:
    """Cross-encoder reranking.

    Worth understanding why this helps rather than just that it does: the
    bi-encoder that produced the index embeds query and passage independently,
    so it can never model their interaction. The cross-encoder reads both
    together and is far more accurate -- and far too slow to run over the whole
    corpus. So it runs over the top-k the cheap retriever already found. That
    two-stage shape is the entire reason reranking is worth its latency.
    """

    name = "cross-encoder"

    def __init__(self, model: str | None = None) -> None:
        s = get_settings()
        self.model_name = model or s.rerank_model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        ce = _load_cross_encoder(self.model_name)
        pairs = [(query, p) for p in passages]
        scores = ce.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
