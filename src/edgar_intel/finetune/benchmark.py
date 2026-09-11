"""Fine-tune vs. prompting benchmark.

The comparison that makes a fine-tuning project worth resume space is not
"I fine-tuned a model and got 89%". It is "I fine-tuned a model, compared it
against a prompting baseline on the same held-out set, and can tell you the
quality, cost and latency tradeoff in both directions."

That requires holding everything constant except the thing being compared:
same held-out examples, same scoring function, same parser. This module does
that and emits a table you can put in a README and defend in an interview.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import get_settings
from ..providers import get_llm
from .dataset import INSTRUCTION, parse_extraction, score_extraction

FEW_SHOT_PREFIX = """Examples:

PASSAGE: Total net sales | 383,285 | 394,328
FACTS: [{"tag":"Revenues","fiscal_year":2023,"value":383285000000,"unit":"USD"}]

PASSAGE: We face intense competition in all of our markets.
FACTS: []

"""


@dataclass(slots=True)
class ArmResult:
    name: str
    n: int
    precision: float
    recall: float
    f1: float
    exact_set_match: float
    p50_latency_ms: int
    p95_latency_ms: int
    total_cost_usd: float
    cost_per_1k_usd: float
    notes: str = ""
    per_case: list[dict[str, Any]] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("per_case")
        return d


def _load_held_out(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            messages = record["messages"]
            rows.append(
                {
                    "passage": messages[1]["content"],
                    "expected": json.loads(messages[2]["content"]),
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def run_arm(
    name: str,
    rows: list[dict[str, Any]],
    model: str | None = None,
    few_shot: bool = False,
    cost_prompt_per_mtok: float | None = None,
    cost_completion_per_mtok: float | None = None,
    notes: str = "",
) -> ArmResult:
    s = get_settings()
    llm = get_llm()

    scores: list[dict[str, float]] = []
    latencies: list[int] = []
    total_cost = 0.0
    per_case: list[dict[str, Any]] = []

    for row in rows:
        prompt = (FEW_SHOT_PREFIX if few_shot else "") + f"PASSAGE: {row['passage']}\nFACTS:"
        started = time.perf_counter()
        completion = llm.complete(
            prompt, system=INSTRUCTION, model=model, temperature=0.0, max_tokens=512
        )
        latency = int((time.perf_counter() - started) * 1000)

        predicted = parse_extraction(completion.text)
        score = score_extraction(predicted, row["expected"])

        p_rate = (
            cost_prompt_per_mtok
            if cost_prompt_per_mtok is not None
            else s.cost_prompt_per_mtok
        )
        c_rate = (
            cost_completion_per_mtok
            if cost_completion_per_mtok is not None
            else s.cost_completion_per_mtok
        )
        cost = (
            completion.prompt_tokens * p_rate + completion.completion_tokens * c_rate
        ) / 1_000_000

        scores.append(score)
        latencies.append(latency)
        total_cost += cost
        per_case.append(
            {
                "predicted": predicted,
                "expected": row["expected"],
                **score,
                "latency_ms": latency,
                "cost_usd": round(cost, 8),
            }
        )

    latencies.sort()
    n = len(rows) or 1
    return ArmResult(
        name=name,
        n=len(rows),
        precision=round(statistics.fmean(s_["precision"] for s_ in scores), 4) if scores else 0.0,
        recall=round(statistics.fmean(s_["recall"] for s_ in scores), 4) if scores else 0.0,
        f1=round(statistics.fmean(s_["f1"] for s_ in scores), 4) if scores else 0.0,
        exact_set_match=round(
            statistics.fmean(s_["exact_set_match"] for s_ in scores), 4
        ) if scores else 0.0,
        p50_latency_ms=int(statistics.median(latencies)) if latencies else 0,
        p95_latency_ms=latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
        if latencies
        else 0,
        total_cost_usd=round(total_cost, 6),
        cost_per_1k_usd=round(total_cost / n * 1000, 4),
        notes=notes,
        per_case=per_case,
    )


def compare(
    held_out_path: str = "data/finetune/held_out.jsonl",
    limit: int = 100,
    arms: list[dict[str, Any]] | None = None,
    out_path: str = "reports/finetune_benchmark.json",
) -> dict[str, Any]:
    """Run every arm over the same held-out rows and emit the comparison.

    Default arms cover the three configurations worth distinguishing. Point
    `base_url` at a local vLLM server serving the merged adapter for the
    fine-tuned arm; the provider abstraction means nothing else changes.
    """
    rows = _load_held_out(held_out_path, limit)
    if not rows:
        raise ValueError(f"no held-out examples found at {held_out_path}")

    arms = arms or [
        {"name": "zero-shot (frontier)", "few_shot": False,
         "notes": "baseline: no examples, no training"},
        {"name": "few-shot (frontier)", "few_shot": True,
         "notes": "two in-context examples; the honest bar to beat"},
    ]

    results = [run_arm(rows=rows, **arm) for arm in arms]
    baseline = results[0]

    table = []
    for r in results:
        row = r.as_row()
        if r is not baseline and baseline.f1:
            row["f1_delta_vs_baseline"] = round(r.f1 - baseline.f1, 4)
            row["cost_ratio_vs_baseline"] = (
                round(r.cost_per_1k_usd / baseline.cost_per_1k_usd, 3)
                if baseline.cost_per_1k_usd
                else None
            )
        table.append(row)

    payload = {
        "held_out_examples": len(rows),
        "arms": table,
        "interpretation": _interpret(results),
    }

    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def _interpret(results: list[ArmResult]) -> str:
    """State the conclusion in the form a hiring manager wants to hear it."""
    if len(results) < 2:
        return "Only one arm was run; there is no comparison to draw."

    best = max(results, key=lambda r: r.f1)
    cheapest = min(results, key=lambda r: r.cost_per_1k_usd)
    baseline = results[0]

    parts = [
        f"Best F1: {best.name} at {best.f1:.3f} "
        f"({best.f1 - baseline.f1:+.3f} vs the {baseline.name} baseline)."
    ]
    if cheapest.name != best.name:
        parts.append(
            f"Cheapest: {cheapest.name} at ${cheapest.cost_per_1k_usd:.4f}/1k requests "
            f"({cheapest.cost_per_1k_usd / baseline.cost_per_1k_usd:.2f}x the baseline) "
            f"for {cheapest.f1:.3f} F1."
        )
        parts.append(
            "The decision is a tradeoff, not a win: quote both numbers and say "
            "which constraint your system is actually under."
        )
    else:
        parts.append(
            f"{best.name} is both the most accurate and the cheapest per request, "
            "so the tradeoff is only against training and operational cost."
        )
    return " ".join(parts)
