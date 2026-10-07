"""Run the golden set through the tool-calling agent that `/ask` serves.

`eval run` measures retrieve-then-answer: hybrid search, then one answering
call with `ANSWER_SYSTEM`. `/ask` runs neither. It runs `run_agent`, which picks
its own tools (XBRL lookups, text search, ratios) for up to eight steps and may
escalate instead of answering. Until this module, no accuracy figure in
docs/METRICS.md measured it.

The comparison is meant to differ in the pipeline and nothing else:

  * the same golden set, loaded the same way;
  * the same grader. An answer goes through `build_result`, exactly as in
    `run_suite`, numeric tolerance and narrative judge included. The judge's
    source context is what the agent read, its tool outputs, just as `eval run`
    passes the passages its answer came from;
  * the same corpus, served index and search defaults, reached through the
    agent's tools.

One rule follows from the pipeline: only what `/ask` returns as an answer is
graded as one. An escalated, exhausted or failed run still returns text, but
it is a handoff. It is scored as declined: not passed, counted with the
abstentions and never as a hallucination, which is how the eval already treats
a model that says the context lacks the figure.

A low-confidence escalation also holds back a draft answer. The draft is graded
on its own and never counts as a pass. It shows whether the confidence floor is
set right: an escalation with a correct draft cost a right answer, and one with
a wrong draft kept a wrong answer from the user.
"""

from __future__ import annotations

import json
import os
import statistics
import uuid
from collections import Counter
from typing import Any

from .. import db
from ..agent.loop import AgentRun, prompt_date, run_agent
from ..agent.tools import SearchFilingsArgs
from ..config import get_settings
from .evidence import index_shape
from .judge import build_result, calibration_pairs, compute_kappa, grade_numeric, kappa_verdict
from .judge_contracts import get_contract
from .runner import (
    INFRA_FAILURE_ABORT_RATE,
    RunAborted,
    _persist_result,
    git_sha,
    is_infra_error,
    summarise_run,
)
from .schemas import CaseResult, EvalCase

# `run_agent`'s own default. Recorded in the config because a ceiling nobody
# wrote down is one nobody can rule out when two runs disagree.
AGENT_COST_CEILING_USD = 0.50

# Outcomes where `/ask` hands the user something other than an answer.
NOT_ANSWERED = ("escalated", "exhausted", "failed", "refused")


def agent_config(persist_traces: bool = False) -> dict[str, Any]:
    """What this run measures: the agent as `/ask` runs it, with every knob."""
    s = get_settings()
    contract = get_contract()
    return {
        # The mark that keeps this run out of `latest_run()`, and so out of
        # `/stats/eval` and `eval gate` (evals/report.py).
        "pipeline": "agent",
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model,
        "judge_model": s.judge_model,
        "judge_contract": contract.name,
        # Given here as the agent's tool outputs, not retrieved passages.
        "judge_sees_source_context": contract.wants_source_context,
        "judge_source_context": "agent tool outputs",
        "agent_max_steps": s.agent_max_steps,
        "agent_max_retries": s.agent_max_retries,
        "agent_confidence_floor": s.agent_confidence_floor,
        "agent_max_tool_output_chars": s.agent_max_tool_output_chars,
        "agent_cost_ceiling_usd": AGENT_COST_CEILING_USD,
        # The system prompt states today's date, so a re-run on another day is
        # a different prompt. Recorded as of the start of the run.
        "agent_prompt_date": prompt_date(),
        # What `search_filings` runs with. The model chooses top_n per call;
        # the schema default applies when it does not.
        "search": {
            "strategy": s.default_strategy,
            "mode": "hybrid",
            "use_rerank": s.use_rerank,
            "k": s.retrieve_k,
            "top_n_default": SearchFilingsArgs.model_fields["top_n"].default,
            "rrf_k": s.rrf_k,
            "item_boost_weight": s.item_boost_weight,
        },
        "embed_provider": s.embed_provider,
        "embed_model": s.embed_model,
        "numeric_tolerance": s.eval_numeric_tolerance,
        "not_answered_scored_as": "declined: not passed, counted as abstention",
        "persist_traces": persist_traces,
    }


def _trace(run: AgentRun) -> list[dict[str, Any]]:
    """The steps, compact: enough to see which tool was called with what."""
    return [
        {
            "n": st.n,
            "kind": st.kind,
            "tool": st.tool,
            "args": st.args,
            "ok": st.ok,
            "note": st.note,
            "result": st.result_summary[:300],
        }
        for st in run.steps
    ]


def grade_agent_run(case: EvalCase, run: AgentRun) -> tuple[CaseResult, dict[str, Any]]:
    """Grade one agent run, and describe what the agent did.

    An answered run goes through `build_result`, the function `run_suite` uses,
    so the two pipelines are marked by the same code.
    """
    s = get_settings()
    evidence = "\n\n".join(run.evidence)
    agent_cost = s.cost_usd(run.prompt_tokens, run.completion_tokens)
    draft_passed: bool | None = None

    if run.outcome == "answered":
        result = build_result(
            case, run.answer, {}, run.total_latency_ms,
            run.prompt_tokens, run.completion_tokens,
            source_context=evidence,
        )
    else:
        # Graded with the numeric grader alone: the narrative judge is not
        # calibrated, and a second judge call per escalation would buy nothing.
        if run.draft_answer and case.kind == "numeric":
            draft_passed, _, _ = grade_numeric(case, run.draft_answer)
        result = CaseResult(
            case_id=case.case_id,
            kind=case.kind,
            passed=False,
            score=0.0,
            answer=run.answer,
            expected=case.expected,
            latency_ms=run.total_latency_ms,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            cost_usd=agent_cost,
            judge_rationale=f"NOT ANSWERED: the agent's outcome was '{run.outcome}'",
            abstained=case.kind == "numeric",
        )

    record = {
        "difficulty": case.difficulty,
        "outcome": run.outcome,
        "confidence": run.confidence,
        "steps": len(run.steps),
        "tools": [st.tool for st in run.steps if st.kind == "tool"],
        "tool_failures": sum(1 for st in run.steps if st.kind == "tool" and not st.ok),
        "errors": [st.note for st in run.steps if st.kind == "error"],
        "draft_answer": run.draft_answer[:1000],
        "draft_passed": draft_passed,
        "agent_cost_usd": round(agent_cost, 8),
        "judge_cost_usd": round(max(0.0, result.cost_usd - agent_cost), 8),
        "evidence_chars": len(evidence),
        "trace_id": run.trace_id,
        "trace": _trace(run),
    }
    return result, record


def _error_record(case: EvalCase, message: str) -> dict[str, Any]:
    return {
        "difficulty": case.difficulty,
        "outcome": "error",
        "confidence": 0.0,
        "steps": 0,
        "tools": [],
        "tool_failures": 0,
        "errors": [message],
        "draft_answer": "",
        "draft_passed": None,
        "agent_cost_usd": 0.0,
        "judge_cost_usd": 0.0,
        "evidence_chars": 0,
        "trace_id": "",
        "trace": [],
    }


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def agent_summary(results: list[CaseResult], records: list[dict[str, Any]]) -> dict[str, Any]:
    """The agent-specific half of a run summary: outcomes, steps, tools, cost."""
    n = len(results)
    outcomes = Counter(r["outcome"] for r in records)
    by_kind: dict[str, dict[str, Any]] = {}
    for result, record in zip(results, records, strict=True):
        key = "narrative" if result.kind == "narrative" else record["difficulty"]
        row = by_kind.setdefault(key, {"n": 0, "passed": 0, "escalated": 0, "not_answered": 0})
        row["n"] += 1
        row["passed"] += bool(result.passed)
        row["escalated"] += record["outcome"] == "escalated"
        row["not_answered"] += record["outcome"] != "answered"
    for row in by_kind.values():
        row["accuracy"] = _rate(row["passed"], row["n"])

    steps = [r["steps"] for r in records if r["outcome"] != "error"] or [0]
    tool_calls = Counter(t for r in records for t in r["tools"])
    cases_using = Counter(t for r in records for t in set(r["tools"]))
    drafts = [r for r in records if r["draft_passed"] is not None]
    agent_cost = sum(r["agent_cost_usd"] for r in records)
    judge_cost = sum(r["judge_cost_usd"] for r in records)
    answered = [
        (res, rec) for res, rec in zip(results, records, strict=True)
        if rec["outcome"] == "answered"
    ]
    return {
        "outcomes": dict(outcomes),
        "answered_rate": _rate(outcomes.get("answered", 0), n),
        "escalation_rate": _rate(outcomes.get("escalated", 0), n),
        "not_answered_rate": _rate(n - outcomes.get("answered", 0), n),
        # Accuracy over the questions the agent chose to answer.
        "accuracy_when_answered": _rate(sum(bool(res.passed) for res, _ in answered), len(answered)),
        "by_kind": by_kind,
        "steps_mean": round(statistics.mean(steps), 2),
        "steps_median": statistics.median(steps),
        "steps_max": max(steps),
        "tool_calls": dict(tool_calls),
        "cases_using_tool": dict(cases_using),
        "tool_failures": sum(r["tool_failures"] for r in records),
        "low_confidence_drafts_graded": len(drafts),
        "low_confidence_drafts_correct": sum(1 for r in drafts if r["draft_passed"]),
        "agent_cost_usd": round(agent_cost, 6),
        "judge_cost_usd": round(judge_cost, 6),
        "agent_cost_per_question_usd": round(agent_cost / max(1, n), 6),
        "errors": outcomes.get("error", 0),
    }


def run_agent_suite(
    cases: list[EvalCase],
    label: str = "",
    sha: str = "",
    progress=None,
    persist_traces: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Run every case through `run_agent`, grade it, and record the run.

    `persist_traces` is off by default: `agent_traces` feeds `/stats/agent`,
    which describes traffic, and a 232-question evaluation is not traffic. The
    steps are kept per case in `eval_results.agent` and in the report.
    """
    s = get_settings()
    sha = sha or git_sha()
    run_key = f"{label or 'agent'}-{uuid.uuid4().hex[:8]}"

    index = index_shape(s.default_strategy)
    if index["chunks"] == 0 or index["embedded"] < index["chunks"]:
        raise RunAborted(
            f"'{s.default_strategy}' has {index['embedded']} of {index['chunks']} chunks "
            "embedded. The agent's search tool would search an empty or partial index."
        )
    config = {**agent_config(persist_traces), "index": index}

    row = db.query_one(
        """
        INSERT INTO eval_runs (run_key, git_sha, label, config, n_cases)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (run_key, sha, label, db.jsonb(config), len(cases)),
    )
    run_id = row["id"]

    results: list[CaseResult] = []
    records: list[dict[str, Any]] = []
    infra_errors = 0
    for i, case in enumerate(cases, start=1):
        evidence = ""
        try:
            run = run_agent(case.question, persist=persist_traces)
            result, record = grade_agent_run(case, run)
            evidence = "\n\n".join(run.evidence)
        except Exception as exc:
            message = str(exc)[:500]
            if is_infra_error(message):
                infra_errors += 1
            result = CaseResult(
                case_id=case.case_id, kind=case.kind, passed=False, score=0.0,
                answer="", expected=case.expected, error=message,
            )
            record = _error_record(case, message)
        results.append(result)
        records.append(record)
        _persist_result(run_id, result, question=case.question, source_context=evidence)
        db.execute(
            "UPDATE eval_results SET agent = %s WHERE run_id = %s AND case_id = %s",
            (db.jsonb(record), run_id, case.case_id),
        )

        # The same void-run rule as `run_suite`: a dead endpoint is not a score.
        if i >= 8 and infra_errors / i > INFRA_FAILURE_ABORT_RATE:
            db.execute(
                "UPDATE eval_runs SET finished_at = now(), summary = %s WHERE id = %s",
                (db.jsonb({"aborted": "infrastructure", "infra_errors": infra_errors}), run_id),
            )
            raise RunAborted(
                f"{infra_errors} of the first {i} cases failed before reaching the "
                f"model ({result.error}). This run is void, not a score of zero."
            )
        if progress:
            progress(i, len(cases), result, record)

    summary = summarise_run(run_key, label, sha, results, config)
    pairs = calibration_pairs(run_id)
    summary.judge_kappa = compute_kappa(pairs)
    summary.judge_labels_matched = len(pairs)
    payload = {**summary.as_dict(), "agent": agent_summary(results, records)}
    db.execute(
        "UPDATE eval_runs SET finished_at = now(), summary = %s WHERE id = %s",
        (db.jsonb(payload), run_id),
    )
    _write_report(payload, results, records)
    return run_id, payload


def _write_report(
    payload: dict[str, Any], results: list[CaseResult], records: list[dict[str, Any]]
) -> str:
    """Every case, not a failure sample: the point is a case-by-case comparison."""
    s = get_settings()
    os.makedirs(s.reports_dir, exist_ok=True)
    path = os.path.join(s.reports_dir, f"{payload['run_key']}.json")
    report = {
        "summary": payload,
        "judge_calibration": kappa_verdict(
            payload.get("judge_kappa"), n_labels_matched=payload.get("judge_labels_matched", 0)
        ),
        "cases": [
            {
                "case_id": res.case_id,
                "kind": res.kind,
                "passed": res.passed,
                "abstained": res.abstained,
                "answer": res.answer[:800],
                "expected": res.expected[:400],
                "why": res.judge_rationale or res.error,
                "latency_ms": res.latency_ms,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "cost_usd": round(res.cost_usd, 8),
                **{k: v for k, v in rec.items() if k != "trace"},
                "trace": rec["trace"],
            }
            for res, rec in zip(results, records, strict=True)
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return path
