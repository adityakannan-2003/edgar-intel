"""Run comparison and the CI regression gate.

The gate is the part of this repo with the clearest production analogue. A
prompt edit, a chunk-size tweak, a model swap: each of these can improve one
thing and quietly degrade another, and none of them are caught by unit tests.
The gate turns "we think this is better" into a check that blocks a merge.

Deliberate design choices:

  * The gate compares against the most recent run carrying the baseline label,
    not against the previous run. Otherwise quality drifts downward one
    tolerable step at a time and every individual step passes.
  * It fails on *any* of overall score, numeric accuracy, or recall@5 dropping
    by more than the tolerance. A change that holds the headline number steady
    while destroying retrieval is a regression.
  * It also fails when the judge's kappa falls below the floor, because at that
    point the narrative half of the score means nothing and passing on it would
    be passing on noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .. import db
from ..config import get_settings
from .judge import KAPPA_FLOOR


@dataclass(slots=True)
class GateResult:
    passed: bool
    reasons: list[str]
    current: dict[str, Any]
    baseline: dict[str, Any] | None
    deltas: dict[str, float]

    def render(self) -> str:
        lines = ["EVAL GATE: " + ("PASS" if self.passed else "FAIL")]
        if self.baseline:
            lines.append(
                f"  baseline: {self.baseline.get('run_key')} "
                f"({self.baseline.get('label')})"
            )
        lines.append(f"  current:  {self.current.get('run_key')}")
        for metric, delta in sorted(self.deltas.items()):
            arrow = "+" if delta >= 0 else ""
            lines.append(f"    {metric:<22} {arrow}{delta:+.4f}")
        for reason in self.reasons:
            lines.append(f"  ! {reason}")
        return "\n".join(lines)


def latest_run(label: str | None = None) -> dict[str, Any] | None:
    sql = """
        SELECT id, run_key, label, git_sha, summary, finished_at
          FROM eval_runs
         WHERE summary IS NOT NULL
    """
    params: list[Any] = []
    if label:
        sql += " AND label = %s"
        params.append(label)
    sql += " ORDER BY finished_at DESC NULLS LAST, id DESC LIMIT 1"
    return db.query_one(sql, params)


def _summary(run: dict[str, Any] | None) -> dict[str, Any]:
    if not run:
        return {}
    summary = run.get("summary")
    if isinstance(summary, str):
        summary = json.loads(summary)
    return summary or {}


GATED_METRICS: tuple[tuple[str, str], ...] = (
    ("overall_score", "overall_score"),
    ("numeric_accuracy", "numeric_accuracy"),
    ("retrieval.recall@5", "recall@5"),
)


def _extract(summary: dict[str, Any], path: str) -> float | None:
    if path.startswith("retrieval."):
        return summary.get("retrieval", {}).get(path.split(".", 1)[1])
    value = summary.get(path)
    return float(value) if isinstance(value, (int, float)) else None


def gate(
    max_regression: float = 0.03,
    baseline_label: str | None = None,
    current_run_key: str | None = None,
) -> GateResult:
    s = get_settings()
    baseline_label = baseline_label or s.eval_baseline_label

    current_run = (
        db.query_one(
            "SELECT id, run_key, label, git_sha, summary FROM eval_runs WHERE run_key = %s",
            (current_run_key,),
        )
        if current_run_key
        else latest_run()
    )
    if not current_run:
        return GateResult(False, ["no evaluation run found"], {}, None, {})

    current = _summary(current_run)
    current["run_key"] = current_run["run_key"]

    baseline_run = latest_run(baseline_label)
    if baseline_run and baseline_run["run_key"] == current_run["run_key"]:
        baseline_run = None

    reasons: list[str] = []
    deltas: dict[str, float] = {}

    kappa = current.get("judge_kappa")
    if kappa is not None and kappa < KAPPA_FLOOR:
        reasons.append(
            f"judge kappa {kappa:.2f} is below the {KAPPA_FLOOR:.2f} floor; "
            "narrative scores are not trustworthy"
        )

    if not baseline_run:
        # First run, or no baseline recorded yet. Not a failure -- but say so,
        # because a gate that silently passes when it has nothing to compare
        # against is worse than no gate.
        reasons.append(
            f"no prior run labelled '{baseline_label}' to compare against; "
            "recording this run as the reference"
        )
        return GateResult(not reasons[:-1], reasons, current, None, {})

    baseline = _summary(baseline_run)
    baseline["run_key"] = baseline_run["run_key"]
    baseline["label"] = baseline_run["label"]

    for path, name in GATED_METRICS:
        cur = _extract(current, path)
        base = _extract(baseline, path)
        if cur is None or base is None:
            continue
        delta = cur - base
        deltas[name] = round(delta, 4)
        if delta < -max_regression:
            reasons.append(
                f"{name} regressed by {abs(delta):.4f} "
                f"({base:.4f} -> {cur:.4f}), tolerance is {max_regression:.4f}"
            )

    # Cost and latency are reported but do not fail the build on their own --
    # a quality win that costs 10ms is usually worth taking, and a hard latency
    # gate belongs in the serving benchmark, not here.
    for key in ("p95_latency_ms", "cost_per_1k_usd"):
        cur, base = current.get(key), baseline.get(key)
        if isinstance(cur, (int, float)) and isinstance(base, (int, float)) and base:
            deltas[key] = round((cur - base) / base, 4)

    hard_failures = [r for r in reasons if "regressed" in r or "kappa" in r]
    return GateResult(not hard_failures, reasons, current, baseline, deltas)


def compare_runs(run_keys: list[str]) -> list[dict[str, Any]]:
    """Side-by-side table rows for a set of runs. Feeds the strategy comparison."""
    rows = db.query(
        """
        SELECT run_key, label, git_sha, summary
          FROM eval_runs
         WHERE run_key = ANY(%s) AND summary IS NOT NULL
         ORDER BY id
        """,
        (run_keys,),
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        summary = _summary(row)
        retrieval = summary.get("retrieval", {})
        out.append(
            {
                "run": row["run_key"],
                "label": row["label"],
                "strategy": summary.get("config", {}).get("strategy"),
                "overall": summary.get("overall_score"),
                "numeric": summary.get("numeric_accuracy"),
                "narrative": summary.get("narrative_pass_rate"),
                "recall@5": retrieval.get("recall@5"),
                "mrr": retrieval.get("mrr"),
                "ndcg@10": retrieval.get("ndcg@10"),
                "p95_ms": summary.get("p95_latency_ms"),
                "cost_per_1k": summary.get("cost_per_1k_usd"),
            }
        )
    return out


def attribute_failure(retrieval: dict[str, Any] | str | None) -> str:
    """'retrieval' when none of the labelled evidence was in the top 5,
    'generation' when some was and the answer still failed, 'unlabelled' when
    the case has no labels to judge by. One rule, shared with `eval regrade`,
    so a re-marked run is attributed exactly as the original was."""
    if isinstance(retrieval, str):
        retrieval = json.loads(retrieval)
    recall = (retrieval or {}).get("recall@5")
    if recall is None:
        return "unlabelled"
    return "retrieval" if recall == 0 else "generation"


def failure_breakdown(run_key: str) -> dict[str, Any]:
    """Where a run actually lost points.

    Splits failures into 'retrieval never found the evidence' and 'evidence was
    retrieved and the model still got it wrong'. These need completely
    different fixes, and conflating them is how people spend a week tuning
    prompts to solve a chunking problem.
    """
    run = db.query_one("SELECT id FROM eval_runs WHERE run_key = %s", (run_key,))
    if not run:
        return {}
    rows = db.query(
        """
        SELECT case_id, kind, passed, retrieval, judge_rationale
          FROM eval_results
         WHERE run_id = %s
        """,
        (run["id"],),
    )

    retrieval_misses = 0
    generation_misses = 0
    unlabelled = 0
    for row in rows:
        if row["passed"]:
            continue
        cause = attribute_failure(row["retrieval"])
        if cause == "unlabelled":
            unlabelled += 1
        elif cause == "retrieval":
            retrieval_misses += 1
        else:
            generation_misses += 1

    total_failures = retrieval_misses + generation_misses + unlabelled
    return {
        "total_cases": len(rows),
        "total_failures": total_failures,
        "retrieval_misses": retrieval_misses,
        "generation_misses": generation_misses,
        "unlabelled_failures": unlabelled,
        "diagnosis": _diagnose(retrieval_misses, generation_misses),
    }


def _diagnose(retrieval_misses: int, generation_misses: int) -> str:
    total = retrieval_misses + generation_misses
    if total == 0:
        return "no gradeable failures"
    share = retrieval_misses / total
    if share >= 0.6:
        return (
            "Most failures are retrieval: the evidence never reached the model. "
            "Work on chunking, hybrid weighting, or k -- prompt changes cannot fix this."
        )
    if share <= 0.3:
        return (
            "Most failures are generation: the evidence was retrieved and the answer was "
            "still wrong. Work on the prompt, output schema, or model choice."
        )
    return "Failures are split roughly evenly between retrieval and generation."
