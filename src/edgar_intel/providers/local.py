"""Local sentence-transformers embedding and cross-encoder reranking.

Both run on CPU. That is a deliberate constraint: it keeps the whole retrieval
side of this project free to run, which means you can iterate on chunking and
fusion as many times as you like without watching a spend meter. Only
generation and judging cost money.
"""

from __future__ import annotations

from functools import lru_cache

from typing import Any

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

    def __init__(self, model: str | None = None, max_length: int | None = None) -> None:
        s = get_settings()
        self.model_name = model or s.rerank_model
        self.max_length = max_length or s.rerank_max_length

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        ce = _load_cross_encoder(self.model_name)
        pairs = [(query, p) for p in passages]
        # Passing max_length explicitly rather than relying on the model default
        # so the truncation point is a stated parameter, not a hidden one.
        scores = ce.predict(pairs, show_progress_bar=False, batch_size=32)
        return [float(s) for s in scores]

    def truncation_report(self, query: str, passages: list[str]) -> dict[str, Any]:
        """How much of each passage the cross-encoder actually reads.

        A reranker that scores worse than no reranker is usually not a bad
        model -- it is a model being shown a fraction of the passage. The
        chunker budgets tokens at roughly four characters each, which holds for
        prose and badly overestimates for financial text: digits, currency
        symbols and table separators tokenize close to one character each. So a
        chunk sized as "512 tokens" can be 800+ real tokens, and everything
        past the limit is invisible to the scorer.

        If a figure sits in the tail of a chunk, the cross-encoder never sees it
        and ranks the correct passage *down*. That is how reranking becomes
        actively harmful rather than merely useless.
        """
        ce = _load_cross_encoder(self.model_name)
        tokenizer = ce.tokenizer
        q_tokens = len(tokenizer.encode(query, add_special_tokens=False))
        budget = self.max_length - q_tokens - 3  # [CLS] and two [SEP]

        lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in passages]
        truncated = [n for n in lengths if n > budget]
        return {
            "query_tokens": q_tokens,
            "passage_budget_tokens": budget,
            "passages": len(passages),
            "truncated": len(truncated),
            "truncated_pct": round(100 * len(truncated) / len(passages), 1) if passages else 0.0,
            "median_passage_tokens": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            "max_passage_tokens": max(lengths) if lengths else 0,
            "median_seen_pct": round(
                100 * min(1.0, budget / (sorted(lengths)[len(lengths) // 2] or 1)), 1
            ) if lengths else 100.0,
        }
