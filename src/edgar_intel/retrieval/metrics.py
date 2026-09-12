"""Retrieval metrics, implemented rather than imported.

These are the numbers that turn "I built a RAG system" into a defensible
claim, so they are written out longhand with tests rather than pulled from a
library -- partly so the definitions are visible, and partly because an
interviewer asking "how is recall@k different from precision@k here" deserves
an answer from someone who wrote the function.

Conventions used throughout:
  * `ranked` is a list of retrieved ids, best first.
  * `relevant` is the set of ids that genuinely answer the query.
  * Every function returns 0.0 for an empty `relevant` set rather than raising,
    because a golden-set case with no labelled evidence is a data bug that
    should show up as a zero in the report, not crash the run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents that appear in the top k.

    This is the metric that actually governs a RAG system's ceiling: if the
    right passage is not in the top k, no amount of prompt engineering
    downstream can recover it. Report it before reporting answer accuracy.
    """
    if not relevant:
        return 0.0
    top = set(ranked[:k])
    return len(top & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k that is relevant. Governs context pollution and cost."""
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for doc in top if doc in relevant) / len(top)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """1/rank of the first relevant hit, 0 if none. Mean over queries is MRR."""
    if not relevant:
        return 0.0
    for idx, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / idx
    return 0.0


def mean_reciprocal_rank(runs: Sequence[tuple[Sequence[str], set[str]]]) -> float:
    if not runs:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in runs) / len(runs)


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain with the standard log2(rank+1) discount."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(
    ranked: Sequence[str],
    relevant: set[str],
    k: int,
    graded: dict[str, float] | None = None,
) -> float:
    """Normalised DCG.

    Binary relevance by default; pass `graded` for per-document gains when some
    evidence is more on-point than the rest. Unlike recall, nDCG is sensitive to
    *where* in the ranking the hits landed, which is what you want once you
    start reranking -- reranking does not change recall@k at all when k is the
    full candidate set, it only moves things up, and nDCG is the metric that
    notices.
    """
    if not relevant:
        return 0.0
    gains = [
        (graded.get(doc, 1.0) if graded else 1.0) if doc in relevant else 0.0
        for doc in ranked
    ]
    actual = dcg_at_k(gains, k)
    ideal_gains = sorted(
        [graded.get(doc, 1.0) if graded else 1.0 for doc in relevant], reverse=True
    )
    ideal = dcg_at_k(ideal_gains, k)
    return actual / ideal if ideal > 0 else 0.0


def hit_rate(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant doc is in the top k.

    Blunt, and for a large part of this evaluation it is the *correct* metric
    while recall@k is not.

    The distinction matters. A numeric case is labelled with every chunk that
    contains the answer -- often five of them, because a figure repeats across
    a table, the MD&A prose and a footnote. The system only has to surface
    *one* to answer the question. recall@5 divides by five and so reports 0.2
    for a retrieval that did its job perfectly; it is measuring redundancy, not
    success.

    Neither replaces the other. recall@k is the honest metric when the task
    genuinely needs all the evidence (a multi-hop comparison across two
    filings). hit@k is the honest metric when any one passage suffices. Both
    are reported so the reader can tell which question is being answered, and
    quoting only the flattering one would be exactly the sort of thing this
    repo exists to avoid.
    """
    if not relevant:
        return 0.0
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def summarise(
    ranked: Sequence[str],
    relevant: set[str],
    ks: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict[str, float]:
    """Everything at once, for one query. Stored per-case in eval_results."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = round(recall_at_k(ranked, relevant, k), 4)
        out[f"precision@{k}"] = round(precision_at_k(ranked, relevant, k), 4)
        out[f"ndcg@{k}"] = round(ndcg_at_k(ranked, relevant, k), 4)
        # Reported alongside recall because for most cases here the two ask
        # different questions and only one of them is the task. See below.
        out[f"hit@{k}"] = round(hit_rate(ranked, relevant, k), 4)
    out["mrr"] = round(reciprocal_rank(ranked, relevant), 4)
    out["first_hit_rank"] = _first_hit_rank(ranked, relevant)
    out["n_relevant"] = float(len(relevant))
    return out


def _first_hit_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for idx, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return float(idx)
    return 0.0


def aggregate(per_case: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean each metric across cases, ignoring keys some cases lack."""
    if not per_case:
        return {}
    keys: set[str] = set()
    for case in per_case:
        keys |= set(case.keys())
    out: dict[str, float] = {}
    for key in sorted(keys):
        vals = [c[key] for c in per_case if key in c]
        if vals:
            out[key] = round(sum(vals) / len(vals), 4)
    return out


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa for two binary raters.

    Used to answer the question nobody asks of their LLM judge: is it actually
    agreeing with a human, or is it agreeing by chance because 90% of answers
    happen to be correct? Raw agreement cannot tell those apart; kappa can.

    Interpretation in common use: <0.20 poor, 0.21-0.40 fair, 0.41-0.60
    moderate, 0.61-0.80 substantial, >0.80 almost perfect. Below ~0.6 the
    judge's narrative scores should not be reported as quality numbers.
    """
    if len(a) != len(b):
        raise ValueError("rater sequences must be the same length")
    n = len(a)
    if n == 0:
        return 0.0

    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n

    a_true = sum(1 for x in a if x) / n
    b_true = sum(1 for y in b if y) / n
    expected = a_true * b_true + (1 - a_true) * (1 - b_true)

    if expected >= 1.0:
        # Both raters were unanimous and identical. Kappa is undefined; report
        # perfect agreement rather than dividing by zero.
        return 1.0
    return round((observed - expected) / (1 - expected), 4)
