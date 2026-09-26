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


def index_label_problem(strategy: str, label: str | None) -> str | None:
    """Why `label` cannot name an index built with `strategy`, or None."""
    from ..chunking import STRATEGIES

    if not label:
        return None
    if label != label.strip() or not label.strip():
        return "an index label cannot be blank or padded with spaces"
    if label in STRATEGIES and label != strategy:
        return (
            f"'{label}' is another strategy's own index; building {strategy} "
            f"under that name would replace it"
        )
    return None


def index_strategy(
    strategy: str,
    target_tokens: int = 512,
    overlap: int = 64,
    rebuild: bool = True,
    progress=None,
    label: str | None = None,
) -> IndexStats:
    """Chunk every filing with `strategy` and embed it, stored as `label`.

    `label` defaults to the strategy name, which replaces that strategy's index.
    Giving another name -- `section_aware_t169` -- builds a second index beside
    it, so a re-chunk can be measured against the index it would replace rather
    than against a number in an old report. Retrieval filters on this name
    only, so `eval run --strategy section_aware_t169` needs nothing else.
    """
    problem = index_label_problem(strategy, label)
    if problem:
        raise ValueError(problem)
    name = label or strategy

    started = time.perf_counter()
    stats = IndexStats(strategy=name)
    embedder = get_embedder()
    s = get_settings()

    if rebuild:
        db.execute("DELETE FROM chunks WHERE strategy = %s", (name,))

    for filing_id in _filing_ids():
        chunks = build_chunks(filing_id, strategy, target_tokens, overlap)
        if not chunks:
            continue
        stats.filings += 1

        rows = [
            (filing_id, c.section_id, name, c.ordinal, estimate_tokens(c.body), c.body)
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
            progress(f"{name}: filing {filing_id} -> {len(rows)} chunks")

    stats.embedded = embed_pending(name, embedder, s.embed_batch, progress)
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


# --------------------------------------------------- embedder window audit
def embed_window_audit(
    strategy: str | None = None, sample: int = 300, seed: int = 7
) -> dict[str, Any]:
    """How much of each chunk the embedding model actually reads.

    `token_est` is `len(body) // 4`, a character heuristic, and the chunkers size
    themselves against a 512-token budget. Neither of those is the embedding
    model's real limit. `all-MiniLM-L6-v2` has a 256 word-piece window and
    `LocalEmbedder.embed` never sets `max_seq_length`, so anything past it is
    dropped before the vector exists -- silently, with the text still present in
    Postgres and still reachable lexically.

    The identical reasoning was already applied to the *reranker*:
    `rerank_max_length` is set explicitly, with a budget for `[CLS]` and two
    `[SEP]`, and a comment about scoring a passage the model has only partly
    read. The embedder got none of it. This function is the measurement that
    turns that suspicion into a number, and it is a command rather than a
    throwaway script because it has to be re-run after any re-chunking.

    Tokenises with the model's own tokeniser. No estimate for the measurement
    itself; the one ratio reported -- word-pieces per `token_est` -- exists to
    turn the finding into the `--target-tokens` a re-chunk needs, because the
    4-characters-per-token rule undercounts figures and table text badly (see
    `LocalReranker.truncation_report`).

    `sample` is per strategy. It used to be one LIMIT across all of them, which
    split 300 rows between four corpora in proportion to their size.
    """
    from ..config import get_settings
    from ..providers.local import _load_encoder

    s = get_settings()
    enc = _load_encoder(s.embed_model)
    window = int(getattr(enc, "max_seq_length", 0) or 0)

    where = "WHERE strategy = %s" if strategy else ""
    params: list[Any] = [str(seed)] + ([strategy] if strategy else []) + [sample]
    rows = db.query(
        f"""
        SELECT strategy, body, token_est
          FROM (
                SELECT strategy, body, token_est,
                       row_number() OVER (
                           PARTITION BY strategy ORDER BY md5(id::text || %s)
                       ) AS rn
                  FROM chunks
                  {where}
               ) sampled
         WHERE rn <= %s
         ORDER BY strategy, rn
        """,
        params,
    )

    tok = enc.tokenizer
    samples: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        # verbose=False: otherwise the tokeniser warns "sequence length is longer
        # than the specified maximum (n > 512)", quoting its own model_max_length
        # -- which is not the 256 the sentence-transformer truncates at.
        n = len(tok.encode(r["body"], add_special_tokens=True, verbose=False))
        token_est = int(r.get("token_est") or estimate_tokens(r["body"]))
        samples.setdefault(r["strategy"], []).append((n, token_est))

    return window_summary(samples, window, model=s.embed_model)


# Chunks shorter than this are left out of the ratio: "Item 1." is 1 token_est
# and 4 word-pieces, and a handful of those would set the p95 on their own.
_RATIO_MIN_TOKEN_EST = 16


def window_summary(
    samples: dict[str, list[tuple[int, int]]], window: int, model: str = ""
) -> dict[str, Any]:
    """Per-strategy truncation figures from (word-pieces, token_est) pairs.

    Word-piece counts include `[CLS]` and `[SEP]`, as `max_seq_length` does.
    `suggested_target_tokens` is the largest `token_est` ceiling at which a
    chunk with the strategy's p95 word-pieces-per-token_est density still fits
    the window -- the number `index build --target-tokens` takes.
    """

    def pct(values: list, q: float):
        if not values:
            return 0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]

    out: list[dict[str, Any]] = []
    for st in sorted(samples):
        pairs = samples[st]
        counts = [n for n, _ in pairs]
        over = [n for n in counts if n > window] if window else []
        # What share of the sampled text the model never sees.
        total = sum(counts)
        seen = sum(min(n, window) for n in counts) if window else total
        ratios = [
            max(0, n - 2) / te for n, te in pairs if te >= _RATIO_MIN_TOKEN_EST
        ]
        p95_ratio = pct(ratios, 0.95) if ratios else 0.0
        suggested = (
            int((window - 2) / p95_ratio) if window and p95_ratio > 0 else None
        )
        out.append(
            {
                "strategy": st,
                "sampled": len(counts),
                "median_tokens": pct(counts, 0.5),
                "p95_tokens": pct(counts, 0.95),
                "max_tokens": max(counts),
                "over_window": len(over),
                "over_window_pct": round(100.0 * len(over) / len(counts), 1),
                "text_unseen_pct": round(100.0 * (1 - seen / total), 1) if total else 0.0,
                "wp_per_token_est_p50": round(pct(ratios, 0.5), 2) if ratios else None,
                "wp_per_token_est_p95": round(p95_ratio, 2) if ratios else None,
                "suggested_target_tokens": suggested,
            }
        )
    suggestions = [r["suggested_target_tokens"] for r in out if r["suggested_target_tokens"]]
    return {
        "model": model,
        "window_tokens": window,
        "rows": out,
        # One ceiling for every strategy, so a re-chunked comparison still
        # changes one variable: the most conservative of the per-strategy fits.
        "suggested_target_tokens": min(suggestions) if suggestions else None,
    }


def embed_window_verdict(audit: dict[str, Any]) -> str:
    """Say what the numbers mean, including when they mean 'no problem'."""
    window = audit["window_tokens"]
    if not window:
        return (
            "Could not read max_seq_length from the model. Without it there is no "
            "window to compare against -- do not conclude anything."
        )
    worst = max(audit["rows"], key=lambda r: r["text_unseen_pct"], default=None)
    if worst is None:
        return "No chunks sampled."
    if worst["text_unseen_pct"] < 1.0:
        return (
            f"Chunks fit the {window}-token window. D8 is not a defect: nothing "
            f"meaningful is being truncated, and the chunking comparison can run."
        )
    verdict = (
        f"{worst['text_unseen_pct']}% of sampled text in `{worst['strategy']}` is "
        f"past the {window}-token window and never reaches the embedder "
        f"({worst['over_window_pct']}% of chunks exceed it; median "
        f"{worst['median_tokens']}, max {worst['max_tokens']}). Dense retrieval is "
        f"scoring those chunks on their opening only. Re-chunk to fit the window "
        f"and re-measure hit@5 before running the strategy comparison -- otherwise "
        f"it ranks four strategies all handicapped the same way."
    )
    target = audit.get("suggested_target_tokens")
    if target:
        verdict += (
            f" A ceiling of about {target} token_est fits the p95 chunk of every "
            f"sampled strategy inside the window: pass it as --target-tokens, and "
            f"re-run this command on the new index to confirm."
        )
    return verdict
