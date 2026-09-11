"""Index building: sections -> chunks -> embeddings -> pgvector.

Written so that the same corpus can be indexed under several chunking
strategies side by side. That is what makes the strategy comparison in the
evaluation honest: identical source text, identical embeddings model,
identical queries, one variable changed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .. import db
from ..chunking import (
    Chunk,
    chunk_fixed,
    chunk_recursive,
    chunk_section_aware,
    chunk_semantic,
    estimate_tokens,
)
from ..config import get_settings
from ..providers import get_embedder


@dataclass(slots=True)
class IndexStats:
    strategy: str
    filings: int = 0
    chunks: int = 0
    embedded: int = 0
    build_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "filings": self.filings,
            "chunks": self.chunks,
            "embedded": self.embedded,
            "build_seconds": round(self.build_seconds, 2),
            "chunks_per_filing": round(self.chunks / self.filings, 1) if self.filings else 0,
        }


def _sections_for(filing_id: int) -> list[dict[str, Any]]:
    return db.query(
        """
        SELECT id, item, title, ordinal, body
          FROM sections
         WHERE filing_id = %s
         ORDER BY ordinal
        """,
        (filing_id,),
    )


def _filing_ids() -> list[int]:
    return [r["id"] for r in db.query("SELECT id FROM filings ORDER BY id")]


def build_chunks(filing_id: int, strategy: str, target_tokens: int, overlap: int) -> list[Chunk]:
    sections = _sections_for(filing_id)
    if not sections:
        return []

    if strategy == "section_aware":
        return chunk_section_aware(sections, target_tokens, overlap)

    # The non-section-aware strategies operate on the concatenated document, so
    # they are free to cut across Item boundaries -- which is exactly the
    # behaviour being measured.
    full_text = "\n\n".join(s["body"] for s in sections)

    if strategy == "fixed":
        pieces = chunk_fixed(full_text, target_tokens, overlap)
    elif strategy == "recursive":
        pieces = chunk_recursive(full_text, target_tokens, overlap)
    elif strategy == "semantic":
        embedder = get_embedder()
        pieces = chunk_semantic(full_text, embedder.embed, target_tokens)
    else:
        raise ValueError(f"unknown chunking strategy: {strategy}")

    return [Chunk(body=p, ordinal=i) for i, p in enumerate(pieces)]


def index_strategy(
    strategy: str,
    target_tokens: int = 512,
    overlap: int = 64,
    rebuild: bool = True,
    progress=None,
) -> IndexStats:
    started = time.perf_counter()
    stats = IndexStats(strategy=strategy)
    embedder = get_embedder()
    s = get_settings()

    if rebuild:
        db.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))

    for filing_id in _filing_ids():
        chunks = build_chunks(filing_id, strategy, target_tokens, overlap)
        if not chunks:
            continue
        stats.filings += 1

        rows = [
            (filing_id, c.section_id, strategy, c.ordinal, estimate_tokens(c.body), c.body)
            for c in chunks
        ]
        db.execute_many(
            """
            INSERT INTO chunks (filing_id, section_id, strategy, ordinal, token_est, body)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (filing_id, strategy, ordinal) DO UPDATE
               SET body = EXCLUDED.body,
                   token_est = EXCLUDED.token_est,
                   section_id = EXCLUDED.section_id
            """,
            rows,
        )
        stats.chunks += len(rows)
        if progress:
            progress(f"{strategy}: filing {filing_id} -> {len(rows)} chunks")

    stats.embedded = embed_pending(strategy, embedder, s.embed_batch, progress)
    stats.build_seconds = time.perf_counter() - started
    return stats


def embed_pending(strategy: str, embedder=None, batch: int = 64, progress=None) -> int:
    """Embed any chunk in this strategy that has no vector yet.

    Separate from chunking so an interrupted embedding run resumes instead of
    restarting -- on a real corpus this is the slow, expensive step.
    """
    embedder = embedder or get_embedder()
    total = 0
    while True:
        rows = db.query(
            """
            SELECT id, body
              FROM chunks
             WHERE strategy = %s AND embedding IS NULL
             ORDER BY id
             LIMIT %s
            """,
            (strategy, batch),
        )
        if not rows:
            break
        vectors = embedder.embed([r["body"] for r in rows])
        db.execute_many(
            "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
            [(vec, row["id"]) for vec, row in zip(vectors, rows, strict=True)],
        )
        total += len(rows)
        if progress:
            progress(f"{strategy}: embedded {total}")
    return total


def index_all(
    strategies: list[str] | None = None,
    target_tokens: int = 512,
    overlap: int = 64,
    progress=None,
) -> list[IndexStats]:
    from ..chunking import STRATEGIES

    strategies = strategies or list(STRATEGIES)
    return [index_strategy(st, target_tokens, overlap, progress=progress) for st in strategies]


def index_report() -> list[dict[str, Any]]:
    """Per-strategy corpus shape. Useful context when comparing recall: a
    strategy producing four times as many chunks has an easier time on recall
    and a harder time on precision, and the report should show that."""
    return db.query(
        """
        SELECT strategy,
               COUNT(*)                       AS chunks,
               COUNT(embedding)               AS embedded,
               ROUND(AVG(token_est))          AS avg_tokens,
               MIN(token_est)                 AS min_tokens,
               MAX(token_est)                 AS max_tokens
          FROM chunks
         GROUP BY strategy
         ORDER BY strategy
        """
    )
