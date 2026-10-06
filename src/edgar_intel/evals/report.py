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
  * In CI the baseline is a committed file, because the database starts empty.
    The file records the exam it was measured on, and a run on any other exam
    fails rather than being compared.
"""

from __future__ import annotations

import hashlib
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
            lines.append(f"    {metric:<22} {delta:+.4f}")
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


# What a golden set asks and expects, not what it is graded against: evidence
# labels are chunk ids, which change whenever the index is rebuilt, and notes
# are working detail. Two sets with the same fingerprint are the same exam.
_EXAM_FIELDS = ("case_id", "kind", "question", "expected", "expected_value", "unit")

# The parts of a run summary a committed baseline keeps. Latency, cost, stage
# timings and the config block depend on the machine and the moment, and would
# make every regenerated baseline a noisy diff; the gate does not fail on them.
_BASELINE_KEYS = (
    "n_cases",
    "numeric_accuracy",
    "narrative_pass_rate",
    "overall_score",
    "abstention_rate",
    "hallucination_rate",
    "retrieval",
)


def golden_fingerprint(cases: list[Any]) -> dict[str, Any]:
    """The exam's identity: its size and a hash of every question and answer."""
    rows = sorted(
        (
            [c.get(f) if isinstance(c, dict) else getattr(c, f) for f in _EXAM_FIELDS]
            for c in cases
        ),
        # Order-free, and never compares a None with a number.
        key=lambda row: json.dumps(row, sort_keys=True),
    )
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()
    return {"n_cases": len(rows), "sha256": digest[:16]}


def save_baseline(run_key: str, golden_path: str, out: str) -> dict[str, Any]:
    """Write a finished run as the reference `gate(baseline_file=...)` reads.

    Refuses a run that did not grade the whole golden set, because the baseline
    would then describe a smaller exam than the one it is checked against.
    """
    from .goldenset import load

    run = db.query_one(
        "SELECT run_key, label, git_sha, summary FROM eval_runs WHERE run_key = %s",
        (run_key,),
    )
    if not run:
        raise ValueError(f"no run {run_key!r}")
    summary = _summary(run)
    exam = golden_fingerprint(load(golden_path))
    if summary.get("n_cases") != exam["n_cases"]:
        raise ValueError(
            f"{run_key} graded {summary.get('n_cases')} cases; "
            f"{golden_path} has {exam['n_cases']}"
        )
    payload = {
        "about": (
            "Reference for `edgar-intel eval gate --baseline-file`. Regenerate it "
            "only when a change is meant to move these numbers; see docs/METRICS.md §4."
        ),
        "run_key": run["run_key"],
        "git_sha": run["git_sha"],
        "golden": exam,
        "summary": {k: summary[k] for k in _BASELINE_KEYS if k in summary},
    }
    retrieval = payload["summary"].get("retrieval")
    if isinstance(retrieval, dict):
        payload["summary"]["retrieval"] = {
            k: v for k, v in retrieval.items() if not k.endswith("_ms")
        }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return payload


def _baseline_from_file(
    path: str, golden_path: str, current: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    """The committed baseline, and any reason it cannot be compared with."""
    from .goldenset import load

    try:
        with open(path, encoding="utf-8") as fh:
            ref = json.load(fh)
        recorded = ref["golden"]
        baseline = dict(ref["summary"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, [f"baseline file {path} is unusable: {exc}"]
    baseline["run_key"] = ref.get("run_key")
    baseline["label"] = path

    try:
        exam = golden_fingerprint(load(golden_path))
    except (OSError, ValueError, TypeError) as exc:
        return None, [f"golden set {golden_path} is unusable: {exc}"]
    problems = []
    if exam != recorded:
        problems.append(
            f"{golden_path} is not the exam the baseline was measured on "
            f"({exam['n_cases']} cases, {exam['sha256']} vs "
            f"{recorded.get('n_cases')} cases, {recorded.get('sha256')}). If the "
            "change is intended, regenerate the baseline (docs/METRICS.md §4)"
        )
    if current.get("n_cases") != exam["n_cases"]:
        problems.append(
            f"the run graded {current.get('n_cases')} cases; "
            f"{golden_path} has {exam['n_cases']}"
        )
    return baseline, problems


def gate(
    max_regression: float = 0.03,
    baseline_label: str | None = None,
    current_run_key: str | None = None,
    baseline_file: str | None = None,
    golden_path: str = "evalset/golden.json",
) -> GateResult:
    """Compare the latest run with a baseline.

    The baseline is the latest database run carrying `baseline_label`, or, with
    `baseline_file`, a committed summary. CI uses the file: its database starts
    empty every time, so a database baseline never exists there, and the gate
    would pass having compared nothing.
    """
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

    reasons: list[str] = []
    blocking: list[str] = []
    deltas: dict[str, float] = {}

    def block(reason: str) -> None:
        reasons.append(reason)
        blocking.append(reason)

    kappa = current.get("judge_kappa")
    if kappa is not None and kappa < KAPPA_FLOOR:
        block(
            f"judge kappa {kappa:.2f} is below the {KAPPA_FLOOR:.2f} floor; "
            "narrative scores are not trustworthy"
        )

    if baseline_file:
        # A file that is missing, malformed, or describes another exam fails
        # the gate. Passing there would be the silent pass this mode exists to
        # remove.
        baseline, problems = _baseline_from_file(baseline_file, golden_path, current)
        for problem in problems:
            block(problem)
        if baseline is None or problems:
            return GateResult(False, reasons, current, baseline, {})
    else:
        baseline_run = latest_run(baseline_label)
        if baseline_run and baseline_run["run_key"] == current_run["run_key"]:
            baseline_run = None
        if not baseline_run:
            # First run, or no baseline recorded yet. Not a failure -- but say
            # so, because a gate that silently passes when it has nothing to
            # compare against is worse than no gate.
            reasons.append(
                f"no prior run labelled '{baseline_label}' to compare against; "
                "recording this run as the reference"
            )
            return GateResult(not blocking, reasons, current, None, {})
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
            block(
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

    return GateResult(not blocking, reasons, current, baseline, deltas)


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
