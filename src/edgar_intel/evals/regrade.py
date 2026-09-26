"""Re-mark a finished run's numeric answers with the current grader. No model calls.

A grader fix changes a run's numbers without changing a single answer, so it
should not cost a paid run to measure -- and a paid run would change the answers
too, making the grader's effect impossible to separate from provider drift.
`eval_results` already stores every answer. This reads them back, grades them
again, and reports every verdict that moved, in both directions.

What it refuses to do is compare across a ground-truth change. The project's
own rule: an accuracy delta that straddles a golden-set rebuild compares two
different exams. So a case whose expected answer no longer matches what the run
stored is skipped and counted, never re-graded against the new expectation.
The run's own numeric tolerance is used, so the grader code is the only thing
that varies.

Read-only: the run's rows are the historical record and stay as they were.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .. import db
from .judge import grade_numeric, is_abstention
from .report import attribute_failure
from .schemas import EvalCase


def default_regrade_name(run_key: str) -> str:
    return os.path.join("reports", f"regrade-{run_key}.json")


def regrade_rows(
    rows: list[dict[str, Any]], cases: dict[str, EvalCase], tolerance: float
) -> dict[str, Any]:
    """The pure half: stored rows in, before/after figures and flips out.

    Only numeric rows are re-marked. Other rows keep their stored verdict and
    are carried only so failure attribution covers the same population as
    `eval failures` did for the original run.
    """
    attribution = {
        "before": {"retrieval": 0, "generation": 0, "unlabelled": 0},
        "after": {"retrieval": 0, "generation": 0, "unlabelled": 0},
    }

    def attribute(when: str, passed: bool, row: dict[str, Any]) -> None:
        if not passed:
            attribution[when][attribute_failure(row.get("retrieval"))] += 1

    numeric_rows = []
    for row in rows:
        if row.get("kind", "numeric") == "numeric":
            numeric_rows.append(row)
        else:
            attribute("before", bool(row.get("passed")), row)
            attribute("after", bool(row.get("passed")), row)

    skipped = {"not_in_golden_set": 0, "expected_changed": 0}
    before = {"passed": 0, "abstained": 0, "wrong": 0}
    after = {"passed": 0, "abstained": 0, "wrong": 0}
    by_difficulty: dict[str, dict[str, int]] = {}
    fail_to_pass: list[dict[str, Any]] = []
    pass_to_fail: list[dict[str, Any]] = []
    graded = 0
    errored = 0

    for row in numeric_rows:
        case = cases.get(row["case_id"])
        if case is None:
            skipped["not_in_golden_set"] += 1
            continue
        if (row.get("expected") or "") != (case.expected or "")[:2000]:
            skipped["expected_changed"] += 1
            continue
        answer = row.get("answer") or ""
        if not answer.strip():
            # An infrastructure error left no answer; the run excluded it from
            # the rates, so does this. It still failed, so it is still attributed.
            errored += 1
            attribute("before", bool(row.get("passed")), row)
            attribute("after", bool(row.get("passed")), row)
            continue

        graded += 1
        abstained = is_abstention(answer)
        was = bool(row.get("passed"))
        now, _, why = grade_numeric(case, answer, tolerance=tolerance)

        before["passed" if was else ("abstained" if abstained else "wrong")] += 1
        after["passed" if now else ("abstained" if abstained else "wrong")] += 1
        attribute("before", was, row)
        attribute("after", now, row)

        bucket = by_difficulty.setdefault(case.difficulty, {"cases": 0, "before": 0, "after": 0})
        bucket["cases"] += 1
        bucket["before"] += int(was)
        bucket["after"] += int(now)

        if was != now:
            flip = {
                "case_id": case.case_id,
                "expected": case.expected,
                "answer": answer[:400],
                "before": row.get("judge_rationale") or "",
                "after": why,
            }
            (fail_to_pass if now else pass_to_fail).append(flip)

    def rates(counts: dict[str, int]) -> dict[str, float | None]:
        if not graded:
            return {"numeric_accuracy": None, "abstention_rate": None, "hallucination_rate": None}
        return {
            "numeric_accuracy": round(counts["passed"] / graded, 4),
            "abstention_rate": round(counts["abstained"] / graded, 4),
            "hallucination_rate": round(counts["wrong"] / graded, 4),
        }

    return {
        "numeric_graded": graded,
        "numeric_errored": errored,
        "skipped": skipped,
        "before": {**before, **rates(before)},
        "after": {**after, **rates(after)},
        "by_difficulty": by_difficulty,
        # Same rule as `eval failures`: a failure with nothing labelled in the
        # top 5 is a retrieval miss, otherwise a generation miss.
        "failure_attribution": attribution,
        "fail_to_pass": fail_to_pass,
        "pass_to_fail": pass_to_fail,
    }


def regrade_run(
    run_key: str,
    cases: list[EvalCase],
    golden_path: str = "",
    out_path: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    from .runner import git_sha

    out_path = out_path or default_regrade_name(run_key)
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Experiment records are evidence; pass "
            "--overwrite to replace it."
        )

    run = db.query_one(
        "SELECT id, run_key, git_sha, config, summary FROM eval_runs WHERE run_key = %s",
        (run_key,),
    )
    if run is None:
        raise ValueError(f"no run with run_key {run_key!r}")
    config = run.get("config") or {}
    summary = run.get("summary") or {}
    if isinstance(config, str):
        config = json.loads(config)
    if isinstance(summary, str):
        summary = json.loads(summary)
    tolerance = float(config.get("numeric_tolerance", 0.005))

    rows = db.query(
        """
        SELECT case_id, kind, passed, answer, expected, judge_rationale, retrieval
          FROM eval_results
         WHERE run_id = %s
         ORDER BY id
        """,
        (run["id"],),
    )
    result = regrade_rows(rows, {c.case_id: c for c in cases}, tolerance)

    reported = summary.get("numeric_accuracy")
    before = result["before"]["numeric_accuracy"]
    payload = {
        "run_key": run_key,
        "run_git_sha": run.get("git_sha"),
        "grader_git_sha": git_sha(),
        "golden_path": golden_path,
        "numeric_tolerance": tolerance,
        "reported_numeric_accuracy": reported,
        # If the stored verdicts do not reproduce the reported figure, the rows
        # are not the run the report describes, and nothing below is comparable.
        "stored_verdicts_reproduce_report": (
            None if reported is None or before is None else abs(reported - before) < 5e-4
        ),
        **result,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    payload["out_path"] = out_path
    return payload
