"""Retrieval-only evaluation sweep.

Answers one question the full eval cannot: *where* is retrieval losing?

The full `eval run` reports a single hit@5 for one configuration. When that
number is mediocre it does not say whether the evidence was never retrieved, or
was retrieved and then buried by the reranker, or was found but cut off by
top_n. Those have opposite fixes -- widen k, retune fusion, drop the reranker --
and picking the wrong one costs a day.

This sweeps configurations over the same golden set and reports retrieval
metrics for each. It never calls an LLM, so it is free and fast: local
embeddings, Postgres full-text, and a local cross-encoder. Run it as often as
you like while tuning.

The number to read first is the **ceiling**: hit@k with reranking off and top_n
set to the full candidate set. That is the best any downstream stage could
possibly do. If the ceiling is high and the shipped config is much lower, the
loss is in ranking or truncation. If the ceiling itself is low, no amount of
reranking will save you and the fix is upstream -- chunking, fusion weights, or
k.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import get_settings
from ..retrieval import metrics as m
from ..retrieval.search import search
from .schemas import EvalCase


@dataclass(slots=True)
class SweepConfig:
    name: str
    mode: str = "hybrid"
    use_rerank: bool = True
    k: int = 50
    top_n: int = 8
    strategy: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SweepResult:
    config: SweepConfig
    n_cases: int
    metrics: dict[str, float] = field(default_factory=dict)
    median_latency_ms: int = 0

    def row(self) -> dict[str, Any]:
        return {
            "config": self.config.name,
            "mode": self.config.mode,
            "rerank": self.config.use_rerank,
            "k": self.config.k,
            "top_n": self.config.top_n,
            "hit@1": self.metrics.get("hit@1"),
            "hit@3": self.metrics.get("hit@3"),
            "hit@5": self.metrics.get("hit@5"),
            "hit@10": self.metrics.get("hit@10"),
            "recall@5": self.metrics.get("recall@5"),
            "mrr": self.metrics.get("mrr"),
            "ndcg@10": self.metrics.get("ndcg@10"),
            "ms": self.median_latency_ms,
        }


def default_sweep(shipped_top_n: int | None = None, shipped_k: int | None = None) -> list[SweepConfig]:
    """The configurations worth comparing, and why each is here.

    Read them as a chain: each isolates one stage so the drop between two rows
    attributes the loss to exactly one thing.
    """
    s = get_settings()
    k = shipped_k or s.retrieve_k
    top_n = shipped_top_n or s.rerank_top_n
    return [
        # What is actually shipped. Everything else is measured against this.
        SweepConfig("shipped", "hybrid", True, k, top_n),
        # Same candidates, no reranking. A drop here means the reranker helps;
        # a rise means it is actively hurting.
        SweepConfig("no-rerank", "hybrid", False, k, top_n),
        # The ceiling: every candidate retrieval found, unranked and untruncated.
        # No downstream stage can beat this.
        SweepConfig("ceiling", "hybrid", False, k, k),
        # Which retriever is carrying the hybrid? If dense alone matches the
        # hybrid, the lexical side is contributing nothing and vice versa.
        SweepConfig("dense-only", "dense", False, k, k),
        SweepConfig("lexical-only", "lexical", False, k, k),
        # Does a wider net help at all, or is the evidence simply unreachable?
        SweepConfig("wide-k", "hybrid", False, k * 4, k * 4),
    ]


def run_config(cases: list[EvalCase], config: SweepConfig, progress=None) -> SweepResult:
    per_case: list[dict[str, float]] = []
    latencies: list[int] = []

    gradeable = [c for c in cases if c.relevant_chunk_ids]
    for i, case in enumerate(gradeable, start=1):
        started = time.perf_counter()
        result = search(
            case.question,
            strategy=config.strategy,
            mode=config.mode,
            use_rerank=config.use_rerank,
            k=config.k,
            top_n=config.top_n,
            ticker=case.ticker,
            fiscal_year=case.fiscal_year,
        )
        latencies.append(int((time.perf_counter() - started) * 1000))
        per_case.append(m.summarise(result.ids, set(case.relevant_chunk_ids)))
        if progress and i % 25 == 0:
            progress(f"{config.name}: {i}/{len(gradeable)}")

    latencies.sort()
    return SweepResult(
        config=config,
        n_cases=len(gradeable),
        metrics=m.aggregate(per_case),
        median_latency_ms=latencies[len(latencies) // 2] if latencies else 0,
    )


def sweep(
    cases: list[EvalCase],
    configs: list[SweepConfig] | None = None,
    out_path: str = "reports/retrieval_sweep.json",
    progress=None,
) -> dict[str, Any]:
    configs = configs or default_sweep()
    results = [run_config(cases, c, progress) for c in configs]

    payload = {
        "n_cases": results[0].n_cases if results else 0,
        "rows": [r.row() for r in results],
        "diagnosis": diagnose(results),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def diagnose(results: list[SweepResult], metric: str = "hit@5") -> str:
    """Say which stage is losing the evidence, in the form of an instruction."""
    by_name = {r.config.name: r for r in results}
    shipped = by_name.get("shipped")
    ceiling = by_name.get("ceiling")
    no_rerank = by_name.get("no-rerank")
    wide = by_name.get("wide-k")
    if not shipped or not ceiling:
        return "sweep incomplete; cannot attribute the loss"

    shipped_v = shipped.metrics.get(metric, 0.0)
    # The ceiling is measured over the whole candidate set, so compare against
    # its hit rate at that depth rather than at 5.
    ceiling_v = max(
        ceiling.metrics.get(f"hit@{k}", 0.0) for k in (1, 3, 5, 10, 20)
    )
    wide_v = (
        max(wide.metrics.get(f"hit@{k}", 0.0) for k in (1, 3, 5, 10, 20))
        if wide
        else ceiling_v
    )

    lines: list[str] = [
        f"shipped {metric}={shipped_v:.3f}; candidate-set ceiling={ceiling_v:.3f}"
    ]

    gap = ceiling_v - shipped_v
    if ceiling_v < 0.6:
        lines.append(
            f"CEILING IS LOW ({ceiling_v:.3f}). The evidence is not in the candidate "
            "set at all, so ranking changes cannot help. Fix upstream: chunking "
            "strategy, fusion weights, or the evidence linker's needles."
        )
        if wide_v > ceiling_v + 0.05:
            lines.append(
                f"Widening k raised it to {wide_v:.3f} -- raise retrieve_k before "
                "anything else."
            )
        else:
            lines.append(
                "Widening k barely moved it, so the passages genuinely do not "
                "match the query. This is a chunking or query-formulation problem."
            )
    elif gap > 0.15:
        lines.append(
            f"Ranking or truncation is losing {gap:.3f}. The evidence IS retrieved "
            "but does not reach the top of the list."
        )
        if no_rerank:
            nr = no_rerank.metrics.get(metric, 0.0)
            if nr > shipped_v + 0.02:
                lines.append(
                    f"Reranking is HURTING: {metric} is {nr:.3f} without it versus "
                    f"{shipped_v:.3f} with it. Check the cross-encoder is scoring "
                    "the right field, then consider dropping it -- it is also the "
                    "slowest stage."
                )
            elif shipped_v > nr + 0.02:
                lines.append(
                    f"Reranking helps ({shipped_v:.3f} vs {nr:.3f}); the remaining "
                    "loss is top_n truncation. Raise rerank_top_n."
                )
            else:
                lines.append(
                    "Reranking is changing almost nothing. It costs the majority of "
                    "retrieval latency, so justify keeping it or remove it."
                )
    else:
        lines.append(
            "Shipped config is close to the ceiling. Further gains have to come "
            "from upstream -- chunking or the retrievers themselves."
        )

    dense = by_name.get("dense-only")
    lexical = by_name.get("lexical-only")
    if dense and lexical:
        d = max(dense.metrics.get(f"hit@{k}", 0.0) for k in (1, 3, 5, 10, 20))
        lx = max(lexical.metrics.get(f"hit@{k}", 0.0) for k in (1, 3, 5, 10, 20))
        lines.append(f"dense-only={d:.3f}, lexical-only={lx:.3f}, hybrid={ceiling_v:.3f}.")
        if ceiling_v <= max(d, lx) + 0.01:
            weaker, stronger = ("lexical", "dense") if d > lx else ("dense", "lexical")
            gap = abs(d - lx)
            if min(d, lx) < 0.05:
                lines.append(
                    f"Fusion adds nothing because the {weaker} retriever returns "
                    "essentially nothing. Check it runs at all before tuning weights."
                )
            elif gap > 0.08:
                lines.append(
                    f"Fusion adds nothing over {stronger} alone. Both retrievers "
                    f"work, but {weaker} is {gap:.3f} weaker and RRF weights them "
                    "equally, so fusing dilutes the stronger list. Weight them by "
                    "measured strength, or drop the weaker one and say why."
                )
            else:
                lines.append(
                    f"Fusion adds nothing over {stronger} alone even though both "
                    "retrievers are comparable -- they are probably returning the "
                    "same documents. Check the overlap before keeping both."
                )
    return " ".join(lines)
