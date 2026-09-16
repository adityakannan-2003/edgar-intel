"""Run the evaluation suite and persist a comparable record of it.

The design constraint that matters: a run has to be reproducible and
comparable. So every run stores the full config it ran under, and every case
stores its own retrieval metrics, latency, tokens and cost. Six months later
you can answer "why was the March run better" instead of guessing.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import time
import uuid
from typing import Any

from .. import db
from ..config import get_settings
from ..providers import get_llm
from ..providers.openai_compat import parse_json_strict
from ..retrieval import metrics as m
from ..retrieval.search import build_context, search
from .judge import build_result, calibration_pairs, compute_kappa, kappa_verdict
from .schemas import ANSWER_SCHEMA, CaseResult, EvalCase, RunSummary

ANSWER_SYSTEM = (
    "You answer questions about SEC filings using only the supplied context. "
    "Cite the bracketed passage numbers you used. If the context does not "
    "contain the answer, say so plainly rather than guessing -- an admission "
    "of missing evidence is correct behaviour, a fabricated figure is not. "
    "Reply with JSON only."
)

ANSWER_TEMPLATE = """CONTEXT:
{context}

QUESTION:
{question}

Answer using only the context above. Return JSON:
{{"answer": "...", "citations": [1, 2], "confidence": 0.0-1.0}}
"""


# Errors that mean the harness never reached the model. A run made entirely of
# these is not a score of zero -- it is an absence of a score, and recording it
# as 0.0 puts a number in the metrics table that describes a missing API key.
_INFRA_ERROR = re.compile(
    r"401 Unauthorized|403 Forbidden|429 Too Many Requests|invalid[_ ]api[_ ]key"
    r"|authentication|insufficient_quota|Connection (?:refused|error|reset)"
    r"|Timeout|Name or service not known|SSL",
    re.I,
)

# Above this share of infrastructure errors the run is void. Not zero-tolerance:
# a handful of rate-limit retries exhausting on a long run is normal and the
# rest of the cases are still meaningful.
INFRA_FAILURE_ABORT_RATE = 0.25


class RunAborted(RuntimeError):
    """Raised when a run failed for reasons that say nothing about quality."""


def is_infra_error(message: str) -> bool:
    return bool(message and _INFRA_ERROR.search(message))


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return ""


def answer_question(
    case: EvalCase,
    strategy: str,
    mode: str = "hybrid",
    use_rerank: bool = True,
    k: int | None = None,
    top_n: int | None = None,
) -> tuple[str, dict[str, Any], int, int, int, Any]:
    """Retrieve, then answer.

    Returns (answer, retrieval_metrics, latency, p_tok, c_tok, retrieval_result).

    The final element is the raw `RetrievalResult`. It exists so a debugging
    harness can inspect exactly which passages reached the model **through this
    same function** rather than re-implementing the pipeline beside it. A probe
    that duplicates the production path can drift from it, and an experiment
    measuring a path nobody ships is worse than no experiment.
    """
    started = time.perf_counter()

    result = search(
        case.question,
        strategy=strategy,
        mode=mode,
        use_rerank=use_rerank,
        k=k,
        top_n=top_n,
        ticker=case.ticker,
        fiscal_year=case.fiscal_year,
    )

    retrieval: dict[str, Any] = {}
    if case.relevant_chunk_ids:
        retrieval = m.summarise(result.ids, set(case.relevant_chunk_ids))
    retrieval["retrieve_ms"] = result.stage_latency_ms.get("retrieve_ms", 0)
    retrieval["rerank_ms"] = result.stage_latency_ms.get("rerank_ms", 0)
    retrieval["n_candidates"] = len(result.hits)

    context = build_context(result.hits)
    completion = get_llm().complete(
        ANSWER_TEMPLATE.format(context=context, question=case.question),
        system=ANSWER_SYSTEM,
        temperature=0.0,
        max_tokens=600,
        json_schema=ANSWER_SCHEMA,
    )

    try:
        payload = parse_json_strict(completion.text)
        answer = str(payload.get("answer", "")).strip()
    except Exception:
        # A model that returns prose instead of JSON still produced an answer;
        # failing the case for a formatting slip would confound schema
        # adherence with factual accuracy, which are different problems.
        answer = completion.text.strip()

    latency_ms = int((time.perf_counter() - started) * 1000)
    return (
        answer,
        retrieval,
        latency_ms,
        completion.prompt_tokens,
        completion.completion_tokens,
        result,
    )


def run_suite(
    cases: list[EvalCase],
    label: str = "",
    strategy: str | None = None,
    mode: str = "hybrid",
    use_rerank: bool = True,
    k: int | None = None,
    top_n: int | None = None,
    sha: str = "",
    progress=None,
) -> tuple[int, RunSummary]:
    s = get_settings()
    strategy = strategy or s.default_strategy
    run_key = f"{label or 'run'}-{uuid.uuid4().hex[:8]}"

    config = {
        "strategy": strategy,
        "mode": mode,
        "use_rerank": use_rerank,
        "k": k or s.retrieve_k,
        "top_n": top_n or s.rerank_top_n,
        "llm_model": s.llm_model,
        "judge_model": s.judge_model,
        "embed_provider": s.embed_provider,
        "embed_model": s.embed_model,
        "llm_provider": s.llm_provider,
        "rrf_k": s.rrf_k,
        "numeric_tolerance": s.eval_numeric_tolerance,
    }

    row = db.query_one(
        """
        INSERT INTO eval_runs (run_key, git_sha, label, config, n_cases)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (run_key, sha or git_sha(), label, db.jsonb(config), len(cases)),
    )
    run_id = row["id"]

    results: list[CaseResult] = []
    infra_errors = 0
    for i, case in enumerate(cases, start=1):
        try:
            answer, retrieval, latency, p_tok, c_tok, _hits = answer_question(
                case, strategy, mode, use_rerank, k, top_n
            )
            result = build_result(case, answer, retrieval, latency, p_tok, c_tok)
        except Exception as exc:
            message = str(exc)[:500]
            if is_infra_error(message):
                infra_errors += 1
            result = CaseResult(
                case_id=case.case_id,
                kind=case.kind,
                passed=False,
                score=0.0,
                answer="",
                expected=case.expected,
                error=message,
            )
        results.append(result)
        _persist_result(run_id, result)

        # Fail fast rather than burning 230 more cases against a dead endpoint.
        # Checked only after a sample large enough to distinguish a bad key from
        # one unlucky timeout.
        if i >= 8 and infra_errors / i > INFRA_FAILURE_ABORT_RATE:
            db.execute(
                "UPDATE eval_runs SET finished_at = now(), summary = %s WHERE id = %s",
                (db.jsonb({"aborted": "infrastructure", "infra_errors": infra_errors}), run_id),
            )
            raise RunAborted(
                f"{infra_errors} of the first {i} cases failed before reaching the "
                f"model ({results[-1].error}). This run is void, not a score of "
                "zero -- no result was recorded. Fix the endpoint or credentials "
                "and re-run."
            )
        if progress:
            progress(i, len(cases), result)

    summary = summarise_run(run_key, label, sha or git_sha(), results, config)
    db.execute(
        """
        UPDATE eval_runs
           SET finished_at = now(), summary = %s
         WHERE id = %s
        """,
        (db.jsonb(summary.as_dict()), run_id),
    )

    kappa = compute_kappa(calibration_pairs(run_id))
    summary.judge_kappa = kappa
    db.execute(
        "UPDATE eval_runs SET summary = %s WHERE id = %s",
        (db.jsonb(summary.as_dict()), run_id),
    )

    _write_report(summary, results, kappa)
    return run_id, summary


def _persist_result(run_id: int, result: CaseResult) -> None:
    db.execute(
        """
        INSERT INTO eval_results (run_id, case_id, kind, passed, score, retrieval,
                                  latency_ms, prompt_tokens, completion_tokens, cost_usd,
                                  answer, expected, judge_rationale)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id, case_id) DO UPDATE
           SET passed = EXCLUDED.passed, score = EXCLUDED.score
        """,
        (
            run_id,
            result.case_id,
            result.kind,
            result.passed,
            result.score,
            db.jsonb(result.retrieval),
            result.latency_ms,
            result.prompt_tokens,
            result.completion_tokens,
            result.cost_usd,
            result.answer[:4000],
            result.expected[:2000],
            (result.judge_rationale or result.error)[:1000],
        ),
    )


def summarise_run(
    run_key: str,
    label: str,
    sha: str,
    results: list[CaseResult],
    config: dict[str, Any],
) -> RunSummary:
    numeric = [r for r in results if r.kind == "numeric"]
    narrative = [r for r in results if r.kind == "narrative"]

    numeric_acc = _mean([1.0 if r.passed else 0.0 for r in numeric])
    narrative_rate = _mean([1.0 if r.passed else 0.0 for r in narrative])

    # Split the numeric failures. A model that abstains when the context lacks
    # the figure is doing what it was asked; a model that returns the wrong
    # figure is not. Cases that errored out are excluded from both -- they
    # measure the harness, not the system.
    graded = [r for r in numeric if not r.error]
    abstained = _mean([1.0 if r.abstained else 0.0 for r in graded])
    hallucinated = _mean(
        [1.0 if (not r.passed and not r.abstained) else 0.0 for r in graded]
    )

    retrieval_dicts = [
        {k: v for k, v in r.retrieval.items() if isinstance(v, (int, float))}
        for r in results
        if r.retrieval
    ]
    retrieval_agg = m.aggregate(retrieval_dicts)

    latencies = sorted(r.latency_ms for r in results) or [0]
    total_cost = sum(r.cost_usd for r in results)

    return RunSummary(
        run_key=run_key,
        label=label,
        git_sha=sha,
        n_cases=len(results),
        numeric_accuracy=round(numeric_acc, 4),
        narrative_pass_rate=round(narrative_rate, 4),
        # Weighted by case count so adding narrative cases cannot quietly swamp
        # the verifiable numeric signal.
        overall_score=round(
            (numeric_acc * len(numeric) + narrative_rate * len(narrative))
            / max(1, len(results)),
            4,
        ),
        judge_kappa=None,
        retrieval=retrieval_agg,
        p50_latency_ms=int(statistics.median(latencies)),
        p95_latency_ms=int(latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]),
        total_cost_usd=round(total_cost, 6),
        cost_per_1k_usd=round(total_cost / max(1, len(results)) * 1000, 4),
        config=config,
        abstention_rate=round(abstained, 4),
        hallucination_rate=round(hallucinated, 4),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


FAILURE_SAMPLE_PER_KIND = 25


def sample_failures_by_kind(
    results: list[CaseResult], per_kind: int = FAILURE_SAMPLE_PER_KIND
) -> list[dict[str, Any]]:
    """Failures grouped by kind, with a quota each.

    The previous version took the first 50 failures in result order. Numeric
    cases are generated before narrative ones, so on a 232-case run the numeric
    failures filled the quota and **all 24 narrative failures were truncated
    away** -- every report showed `narrative_pass_rate: 0.0` and not one example
    of why.

    That hid a real defect for four runs. The narrative judge was returning HTTP
    400 on every call; the report had room to show it and did not. A reporting
    cap that can silently drop an entire case kind is not a display detail, it
    is a hole in the instrumentation.

    Quota per kind rather than a global cap, so a kind can never be crowded out
    by a noisier one. Order within a kind is preserved.
    """
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        if r.passed:
            continue
        rows = by_kind.setdefault(r.kind, [])
        if len(rows) >= per_kind:
            continue
        rows.append(
            {
                "case_id": r.case_id,
                "kind": r.kind,
                "expected": r.expected[:400],
                "answer": r.answer[:400],
                "why": r.judge_rationale or r.error,
            }
        )
    return [row for kind in sorted(by_kind) for row in by_kind[kind]]


def failure_counts_by_kind(results: list[CaseResult]) -> dict[str, int]:
    """Total failures per kind, so the sample never has to be mistaken for all of them."""
    counts: dict[str, int] = {}
    for r in results:
        if not r.passed:
            counts[r.kind] = counts.get(r.kind, 0) + 1
    return counts


def _write_report(summary: RunSummary, results: list[CaseResult], kappa: float | None) -> None:
    s = get_settings()
    os.makedirs(s.reports_dir, exist_ok=True)
    path = os.path.join(s.reports_dir, f"{summary.run_key}.json")
    payload = {
        "summary": summary.as_dict(),
        "judge_calibration": kappa_verdict(kappa),
        "failure_counts": failure_counts_by_kind(results),
        "failure_sample_per_kind": FAILURE_SAMPLE_PER_KIND,
        "failures": sample_failures_by_kind(results),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def compare_strategies(
    cases: list[EvalCase],
    strategies: list[str],
    label_prefix: str = "strategy",
    progress=None,
) -> list[RunSummary]:
    """Run the same cases under each chunking strategy.

    This is the experiment that produces the comparison table -- and the
    resume bullet. Same questions, same models, same judge; one variable.
    """
    summaries: list[RunSummary] = []
    for strategy in strategies:
        _, summary = run_suite(
            cases, label=f"{label_prefix}-{strategy}", strategy=strategy, progress=progress
        )
        summaries.append(summary)
    return summaries
