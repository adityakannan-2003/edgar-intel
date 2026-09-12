"""Hybrid retrieval: dense vectors + lexical search, fused with RRF.

Why hybrid rather than pure vector search, stated plainly because it is the
most common thing candidates get asked to justify:

Dense retrieval generalises. It finds "the company's borrowing costs rose" when
you asked about interest expense. It is also systematically bad at exactly the
tokens that matter most in a filing -- ticker symbols, tag names, specific
dollar amounts, years -- because an embedding of a 512-token passage does not
preserve a single rare literal well. Lexical search is the mirror image: exact
on literals, useless on paraphrase.

On financial filings you need both, because a real question ("what did NVDA
report for R&D in fiscal 2024") contains a rare literal *and* a paraphrase.

Fusion is Reciprocal Rank Fusion rather than score normalisation. Cosine
distances and ts_rank_cd scores live on incomparable scales, and any attempt to
normalise them into a weighted sum needs constants that have to be re-tuned
whenever either retriever changes. RRF only uses rank position, so it needs no
calibration and degrades gracefully.

An honest note about "BM25": the lexical side here is Postgres full-text search
scored with ts_rank_cd, which is a tf-idf-family ranking but is NOT BM25 -- it
has no document-length saturation term. Calling it BM25 would be wrong. If you
need true BM25 in Postgres, ParadeDB's pg_search provides it; swapping it in is
a change to `lexical_search` and nothing else.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .. import db
from ..config import get_settings
from ..providers import get_embedder, get_reranker

_WORD = re.compile(r"[A-Za-z0-9]+")


def to_or_tsquery(query: str, max_terms: int = 24) -> str:
    """Build an OR-ed tsquery from a natural-language question.

    This exists because of a Postgres behaviour that fails silently and cost
    this project its entire lexical retriever.

    `websearch_to_tsquery` joins unquoted words with AND. So the question

        "What was Caterpillar's total revenue for fiscal year 2024?"

    becomes 'caterpillar' & 'total' & 'revenu' & 'fiscal' & 'year' & '2024' --
    every lexeme required in a single chunk. No chunk satisfies that, so the
    query returns zero rows. Not an error, not a warning: an empty result, on
    every query, which made the hybrid retriever silently dense-only and the
    RRF fusion a no-op. A retrieval sweep is what surfaced it.

    OR-ing the terms is the right shape for retrieval. Recall comes from the
    match, precision comes from `ts_rank_cd`, which scores by how many query
    lexemes a document contains and how densely they cluster. Matching broadly
    and ranking well is the whole idea behind a lexical retriever; requiring
    every term is a filter, not a search.

    Stopwords need no special handling -- the 'english' configuration strips
    them from both the tsvector and the tsquery.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for raw in _WORD.findall(query):
        term = raw.lower()
        # Single characters carry no signal and bloat the query.
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return " | ".join(terms)


@dataclass(slots=True)
class Hit:
    chunk_id: str
    body: str
    filing_id: int
    score: float
    source: str                       # dense | lexical | fused | reranked
    cik: str | None = None
    ticker: str | None = None
    fiscal_year: int | None = None
    item: str | None = None
    ranks: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "source": self.source,
            "ticker": self.ticker,
            "fiscal_year": self.fiscal_year,
            "item": self.item,
            "ranks": self.ranks,
        }


@dataclass(slots=True)
class RetrievalResult:
    hits: list[Hit]
    latency_ms: int
    stage_latency_ms: dict[str, int] = field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        return [h.chunk_id for h in self.hits]


_BASE_SELECT = """
    SELECT c.id::text AS chunk_id,
           c.body,
           c.filing_id,
           c.section_id,
           s.item,
           f.cik,
           f.fiscal_year,
           co.ticker
      FROM chunks c
      JOIN filings f  ON f.id = c.filing_id
      JOIN companies co ON co.cik = f.cik
      LEFT JOIN sections s ON s.id = c.section_id
"""


def _filters(
    ticker: str | None,
    fiscal_year: int | None,
    item: str | None,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if ticker:
        clauses.append("co.ticker = %s")
        params.append(ticker.upper())
    if fiscal_year:
        clauses.append("f.fiscal_year = %s")
        params.append(fiscal_year)
    if item:
        clauses.append("s.item = %s")
        params.append(item)
    return (" AND " + " AND ".join(clauses) if clauses else "", params)


def dense_search(
    query: str,
    strategy: str | None = None,
    k: int | None = None,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
) -> list[Hit]:
    s = get_settings()
    strategy = strategy or s.default_strategy
    k = k or s.retrieve_k

    vec = get_embedder().embed([query])[0]
    where, params = _filters(ticker, fiscal_year, item)

    # `<=>` is pgvector's cosine distance: 0 identical, 2 opposite. Converting
    # to a similarity keeps "higher is better" consistent across retrievers.
    sql = f"""
        {_BASE_SELECT}
         WHERE c.strategy = %s AND c.embedding IS NOT NULL{where}
         ORDER BY c.embedding <=> %s::vector
         LIMIT %s
    """
    rows = db.query(sql, [strategy, *params, vec, k])
    return [
        Hit(
            chunk_id=r["chunk_id"],
            body=r["body"],
            filing_id=r["filing_id"],
            score=1.0,  # replaced by rank-based fusion; raw distance is not comparable
            source="dense",
            cik=r["cik"],
            ticker=r["ticker"],
            fiscal_year=r["fiscal_year"],
            item=r["item"],
            ranks={"dense": i + 1},
        )
        for i, r in enumerate(rows)
    ]


def lexical_search(
    query: str,
    strategy: str | None = None,
    k: int | None = None,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
) -> list[Hit]:
    s = get_settings()
    strategy = strategy or s.default_strategy
    k = k or s.retrieve_k
    where, params = _filters(ticker, fiscal_year, item)

    tsquery = to_or_tsquery(query)
    if not tsquery:
        # Nothing lexically searchable (punctuation or stopwords only).
        return []

    sql = f"""
        {_BASE_SELECT}
         WHERE c.strategy = %s
           AND c.body_tsv @@ to_tsquery('english', %s){where}
         ORDER BY ts_rank_cd(c.body_tsv, to_tsquery('english', %s)) DESC
         LIMIT %s
    """
    rows = db.query(sql, [strategy, tsquery, *params, tsquery, k])
    return [
        Hit(
            chunk_id=r["chunk_id"],
            body=r["body"],
            filing_id=r["filing_id"],
            score=1.0,
            source="lexical",
            cik=r["cik"],
            ticker=r["ticker"],
            fiscal_year=r["fiscal_year"],
            item=r["item"],
            ranks={"lexical": i + 1},
        )
        for i, r in enumerate(rows)
    ]


def reciprocal_rank_fusion(
    rankings: list[list[Hit]],
    weights: list[float] | None = None,
    rrf_k: int | None = None,
) -> list[Hit]:
    """Fuse ranked lists by sum of weight / (rrf_k + rank).

    rrf_k damps the influence of the very top positions so one retriever being
    confidently wrong cannot dominate the fused list. 60 is the constant from
    the original paper and is left untuned on purpose -- tuning it against the
    same evaluation set used for reporting would leak the test set into the
    system.
    """
    s = get_settings()
    rrf_k = rrf_k or s.rrf_k
    weights = weights or [1.0] * len(rankings)

    scores: dict[str, float] = {}
    merged: dict[str, Hit] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (rrf_k + rank)
            if hit.chunk_id in merged:
                merged[hit.chunk_id].ranks.update(hit.ranks)
            else:
                merged[hit.chunk_id] = hit

    fused: list[Hit] = []
    for chunk_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        hit = merged[chunk_id]
        hit.score = score
        hit.source = "fused"
        fused.append(hit)
    return fused


def rerank(query: str, hits: list[Hit], top_n: int | None = None) -> list[Hit]:
    """Cross-encoder rerank of the fused candidates.

    This is the expensive stage -- one forward pass per (query, passage) pair --
    which is precisely why it runs over ~50 candidates and not the corpus. Time
    it separately from retrieval so the latency budget stays legible.
    """
    s = get_settings()
    top_n = top_n or s.rerank_top_n
    if not hits:
        return []
    scores = get_reranker().score(query, [h.body for h in hits])
    for hit, score in zip(hits, scores, strict=True):
        hit.score = float(score)
        hit.source = "reranked"
    ordered = sorted(hits, key=lambda h: h.score, reverse=True)
    for rank, hit in enumerate(ordered, start=1):
        hit.ranks["reranked"] = rank
    return ordered[:top_n]


def search(
    query: str,
    strategy: str | None = None,
    k: int | None = None,
    top_n: int | None = None,
    mode: str = "hybrid",
    use_rerank: bool = True,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
) -> RetrievalResult:
    """The retrieval entry point. `mode` is one of dense | lexical | hybrid."""
    s = get_settings()
    started = time.perf_counter()
    stage: dict[str, int] = {}

    t0 = time.perf_counter()
    if mode == "dense":
        candidates = dense_search(query, strategy, k, ticker, fiscal_year, item)
    elif mode == "lexical":
        candidates = lexical_search(query, strategy, k, ticker, fiscal_year, item)
    elif mode == "hybrid":
        dense = dense_search(query, strategy, k, ticker, fiscal_year, item)
        lexical = lexical_search(query, strategy, k, ticker, fiscal_year, item)
        candidates = reciprocal_rank_fusion(
            [dense, lexical], weights=[s.dense_weight, s.lexical_weight]
        )
    else:
        raise ValueError(f"unknown retrieval mode: {mode}")
    stage["retrieve_ms"] = int((time.perf_counter() - t0) * 1000)

    if use_rerank and candidates:
        t1 = time.perf_counter()
        candidates = rerank(query, candidates, top_n)
        stage["rerank_ms"] = int((time.perf_counter() - t1) * 1000)
    else:
        candidates = candidates[: (top_n or s.rerank_top_n)]

    return RetrievalResult(
        hits=candidates,
        latency_ms=int((time.perf_counter() - started) * 1000),
        stage_latency_ms=stage,
    )


def build_context(hits: list[Hit], max_chars: int = 12000) -> str:
    """Assemble retrieved passages into a labelled context block.

    Every passage is tagged with its source so the generated answer can cite,
    and so a wrong answer can be traced to the passage that caused it. An
    untraceable answer is an undebuggable one.
    """
    parts: list[str] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        tag = f"[{i}] {hit.ticker or hit.cik} FY{hit.fiscal_year or '?'}"
        if hit.item:
            tag += f" Item {hit.item}"
        block = f"{tag}\n{hit.body}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)
