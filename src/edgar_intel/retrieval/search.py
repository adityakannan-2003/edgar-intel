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
    """What retrieval produced, before and after `top_n` truncation.

    `candidates` carries the full ordering the ranker produced; `hits` is that
    list cut to `top_n`. Both come from **one** `search()` call, which is the
    point: a probe that ran retrieval twice -- once for diagnostics, once for
    generation -- silently measured two different configurations. The item boost
    reached the diagnostic call and not the generating one, so the reported
    pre-truncation ranks improved to [3, 4, 11, 13, 15] while the answer was
    still being produced from the unboosted [9, 11, 13, 21, 29]. Every arm of
    that experiment was void.

    One call, two views. It is not possible to configure them differently.
    """

    hits: list[Hit]
    latency_ms: int
    stage_latency_ms: dict[str, int] = field(default_factory=dict)
    candidates: list[Hit] = field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        return [h.chunk_id for h in self.hits]

    @property
    def candidate_ids(self) -> list[str]:
        return [h.chunk_id for h in self.candidates]

    def ranks_of(self, chunk_ids: set[str], *, pre_truncation: bool = False) -> list[int]:
        """1-indexed positions of the given chunks, in either view."""
        source = self.candidates if pre_truncation else self.hits
        return [i for i, h in enumerate(source, start=1) if h.chunk_id in chunk_ids]


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


# Which 10-K Item answers which kind of question.
#
# This is a property of Form 10-K, not of any company: the SEC prescribes what
# goes in each Item, so "what legal proceedings are disclosed" is answered in
# Item 3 for every filer that has ever filed one. Nothing here names a company,
# a year, or a case.
#
# Cues are matched against the question only. They are deliberately the words a
# person would use, not the Item titles, because a question that already said
# "Item 3" would not need this.
ITEM_QUESTION_CUES: list[tuple[str, tuple[str, ...]]] = [
    ("3", ("legal proceeding", "lawsuit", "litigation", "court", "antitrust",
           "regulatory proceeding", "sued", "settlement")),
    ("1A", ("risk factor", "risks", "risk", "supplier concentration", "dependence on",
            "could adversely", "vulnerab", "competitive pressure", "competition")),
    ("7A", ("market risk", "foreign currency", "exchange rate", "interest rate",
            "hedg", "derivative", "commodity price")),
    ("7", ("md&a", "management's discussion", "drivers of", "results of operations",
           "revenue change", "gross margin", "liquidity", "cash flow")),
    ("8", ("financial statement", "balance sheet", "income statement", "footnote",
           "accounting polic")),
    ("1", ("business", "products", "segments", "customers", "employees",
           "supply chain", "manufactur", "distribution")),
    ("2", ("propert", "facilities", "headquarters", "real estate")),
    ("9A", ("internal control", "disclosure controls", "material weakness")),
]


ITEM_ORDER = {item: i for i, (item, _) in enumerate(ITEM_QUESTION_CUES)}


def infer_expected_items(query: str) -> list[str]:
    """Which 10-K Items a question is asking about, best first.

    Returns an empty list when nothing matches, which is the common case for the
    numeric questions -- a figure can legitimately appear in Item 7, Item 8 or a
    footnote, so guessing would hurt.

    Why a list and not one Item: `incorporation by reference` is routine. NVIDIA's
    Item 8 is three sentences pointing at the financial statements; P&G's Item 7A
    points into the MD&A. This project has already been bitten by treating a
    short Item as a missing one, so the ranking signal must tolerate the content
    living somewhere else.
    """
    lowered = query.lower()
    matched: list[tuple[int, str]] = []
    for item, cues in ITEM_QUESTION_CUES:
        hits = sum(1 for cue in cues if cue in lowered)
        if hits:
            matched.append((hits, item))
    matched.sort(key=lambda pair: (-pair[0], ITEM_ORDER.get(pair[1], 99)))
    return [item for _, item in matched]


def item_ranking(hits: list[Hit], items: list[str]) -> list[Hit]:
    """The candidates that sit in an expected Item, in their existing order.

    Fed into RRF as an additional ranked list rather than applied as a score
    multiplier, so the signal stays rank-based and needs no calibration against
    cosine distances or `ts_rank_cd` scores -- the same reason fusion is RRF in
    the first place.

    A *boost*, never a filter. Filtering to Item 3 would have scored well on the
    Apple legal case and destroyed every question whose evidence is incorporated
    by reference from another Item.
    """
    if not items:
        return []
    priority = {item: i for i, item in enumerate(items)}
    scoped = [(priority[h.item], i, h) for i, h in enumerate(hits) if h.item in priority]
    # Ties broken by the incoming fused order, so this list adds Item
    # information and nothing else -- it never reorders within an Item.
    scoped.sort(key=lambda triple: (triple[0], triple[1]))
    return [h for _, _, h in scoped]


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    """Cross-encoder rerank of the fused candidates. Returns the FULL ordering.

    This is the expensive stage -- one forward pass per (query, passage) pair --
    which is precisely why it runs over ~50 candidates and not the corpus. Time
    it separately from retrieval so the latency budget stays legible.

    Truncation used to happen here, which put two cuts in the pipeline: this one
    and the `top_n` slice in `search()`. Two places that shorten a list are two
    places to look when something vanishes. Ranking happens here, cutting
    happens in `search()`, and the uncut ordering survives on
    `RetrievalResult.candidates` so a probe can see what was thrown away.
    """
    if not hits:
        return []
    scores = get_reranker().score(query, [h.body for h in hits])
    for hit, score in zip(hits, scores, strict=True):
        hit.score = float(score)
        hit.source = "reranked"
    ordered = sorted(hits, key=lambda h: h.score, reverse=True)
    for rank, hit in enumerate(ordered, start=1):
        hit.ranks["reranked"] = rank
    return ordered


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
    item_boost: float | None = None,
) -> RetrievalResult:
    """The retrieval entry point. `mode` is one of dense | lexical | hybrid.

    `item_boost` weights an additional RRF list of candidates that sit in the
    10-K Item the question is about. 0.0 disables it, which is the default --
    it must be measured before it ships, and a default-on ranking change would
    contaminate every A/B comparison already in flight.
    """
    s = get_settings()
    started = time.perf_counter()
    stage: dict[str, int] = {}
    boost = s.item_boost_weight if item_boost is None else item_boost

    t0 = time.perf_counter()
    if mode == "dense":
        candidates = dense_search(query, strategy, k, ticker, fiscal_year, item)
    elif mode == "lexical":
        candidates = lexical_search(query, strategy, k, ticker, fiscal_year, item)
    elif mode == "hybrid":
        dense = dense_search(query, strategy, k, ticker, fiscal_year, item)
        lexical = lexical_search(query, strategy, k, ticker, fiscal_year, item)
        rankings = [dense, lexical]
        weights = [s.dense_weight, s.lexical_weight]
        if boost > 0:
            fused_order = reciprocal_rank_fusion([dense, lexical], weights=weights)
            scoped = item_ranking(fused_order, infer_expected_items(query))
            if scoped:
                rankings.append(scoped)
                weights.append(boost)
        candidates = reciprocal_rank_fusion(rankings, weights=weights)
    else:
        raise ValueError(f"unknown retrieval mode: {mode}")
    stage["retrieve_ms"] = int((time.perf_counter() - t0) * 1000)

    if use_rerank and candidates:
        t1 = time.perf_counter()
        candidates = rerank(query, candidates)
        stage["rerank_ms"] = int((time.perf_counter() - t1) * 1000)

    # The single truncation point in the pipeline.
    cut = top_n or s.rerank_top_n
    return RetrievalResult(
        hits=candidates[:cut],
        latency_ms=int((time.perf_counter() - started) * 1000),
        stage_latency_ms=stage,
        candidates=candidates,
    )


DEFAULT_CONTEXT_MAX_CHARS = 12000


@dataclass(slots=True)
class ContextReport:
    """What actually reached the prompt, as opposed to what retrieval selected.

    `result.hits` is not the context the model saw. Between the two sits a
    character cap that can drop passages silently, and conflating them is how a
    `top_n` experiment produces four arms with identical prompt-token counts and
    nobody notices for a day.
    """

    text: str
    selected_ids: list[str] = field(default_factory=list)
    included_ids: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)
    chars_before_cap: int = 0
    chars_after_cap: int = 0
    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS
    first_excluded_id: str | None = None
    first_excluded_rank: int | None = None
    first_excluded_chars: int = 0
    packing: str = "greedy-stop"
    would_fit_with_skip: int = 0

    @property
    def n_selected(self) -> int:
        return len(self.selected_ids)

    @property
    def n_included(self) -> int:
        return len(self.included_ids)

    @property
    def cap_binding(self) -> bool:
        return bool(self.dropped_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_selected": self.n_selected,
            "n_included": self.n_included,
            "included_ids": self.included_ids,
            "dropped_ids": self.dropped_ids,
            "chars_before_cap": self.chars_before_cap,
            "chars_after_cap": self.chars_after_cap,
            "max_chars": self.max_chars,
            "cap_binding": self.cap_binding,
            "first_excluded_id": self.first_excluded_id,
            "first_excluded_rank": self.first_excluded_rank,
            "first_excluded_chars": self.first_excluded_chars,
            "packing": self.packing,
            "would_fit_with_skip": self.would_fit_with_skip,
        }


def build_context_report(
    hits: list[Hit],
    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    packing: str = "greedy-stop",
) -> ContextReport:
    """Assemble the context block and account for every passage that did not make it.

    Every passage is tagged with its source so the generated answer can cite,
    and so a wrong answer can be traced to the passage that caused it. An
    untraceable answer is an undebuggable one -- and a passage that vanished
    without being counted is worse still.

    Two packing policies, because the shipped one has a sharp edge:

    `greedy-stop` (the shipped behaviour, kept as the default) stops at the
    first passage that does not fit. One long passage at rank 4 therefore
    discards ranks 5 through `top_n` **even when they would have fit**. That is
    not a cap, it is a cliff, and it is the reason prompt tokens sat flat near
    3,000 whether `top_n` was 8 or 20: the loop was ending in the same place
    every time, so raising `top_n` bought nothing at all.

    `skip-oversized` keeps going and takes later passages that still fit.

    The default is unchanged on purpose. Changing what the model reads while
    also measuring `top_n` would confound the experiment; `would_fit_with_skip`
    reports what the other policy would have admitted, so the choice can be made
    from a number rather than from taste.
    """
    blocks: list[tuple[Hit, str]] = []
    for i, hit in enumerate(hits, start=1):
        tag = f"[{i}] {hit.ticker or hit.cik} FY{hit.fiscal_year or '?'}"
        if hit.item:
            tag += f" Item {hit.item}"
        blocks.append((hit, f"{tag}\n{hit.body}"))

    report = ContextReport(
        text="",
        selected_ids=[h.chunk_id for h in hits],
        chars_before_cap=sum(len(b) for _, b in blocks),
        max_chars=max_chars,
        packing=packing,
    )

    parts: list[str] = []
    used = 0
    stopped = False
    for rank, (hit, block) in enumerate(blocks, start=1):
        fits = used + len(block) <= max_chars
        if stopped or not fits:
            if report.first_excluded_id is None:
                report.first_excluded_id = hit.chunk_id
                report.first_excluded_rank = rank
                report.first_excluded_chars = len(block)
            if packing == "greedy-stop":
                stopped = True
            report.dropped_ids.append(hit.chunk_id)
            continue
        parts.append(block)
        report.included_ids.append(hit.chunk_id)
        used += len(block)

    report.text = "\n\n".join(parts)
    report.chars_after_cap = used

    # What a skip-oversized policy would have admitted, measured without
    # changing what this call returns.
    skip_used, skip_n = 0, 0
    for _, block in blocks:
        if skip_used + len(block) <= max_chars:
            skip_used += len(block)
            skip_n += 1
    report.would_fit_with_skip = skip_n
    return report


def build_context(
    hits: list[Hit],
    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    packing: str = "greedy-stop",
) -> str:
    """The context string alone, for callers that do not need the accounting."""
    return build_context_report(hits, max_chars, packing).text
