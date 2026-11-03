"""Model routing for the agent `/ask` serves: which model answers which question.

Three policies, measured on the golden set with the grader `eval agent` uses
(`grade_agent_run`), so the arms differ in the model and nothing else
(docs/ROUTER.md):

  single:<model>               every question to one model.
  cascade:<cheap>,<strong>     the cheap model first. When its run ends without
                               an answer -- it escalated, a citation or
                               grounding check rejected its answer past the
                               retry budget, it hit the step ceiling or it never
                               produced valid JSON -- the strong model answers
                               from scratch.
  difficulty:<cheap>,<strong>  the cheap model rates the question easy or hard
                               in one short call. Easy goes to the cheap model;
                               hard, or a rating that did not parse, to the
                               strong one.

Every model call is priced from its own reported tokens at list price, cached
input included (`finetune.benchmark.PRICES_USD_PER_MTOK`). The configured
`cost_usd` knows one model's prices, and this compares four.

Like the fine-tune benchmark, a run appends one JSON line per case as it
finishes, so an interrupted paid run keeps what it bought. Re-running the same
command resumes and retries only the cases that errored. `score` reads the
files and makes no model call.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..agent.loop import AgentRun, run_agent
from ..config import get_settings
from ..finetune.benchmark import _OPENAI_REASONING, PRICES_USD_PER_MTOK, _percentile
from ..providers import LLMProvider, get_llm
from ..providers.base import Completion
from ..providers.openai_compat import parse_json_strict
from .agent_eval import agent_config, grade_agent_run
from .evidence import index_shape
from .runner import INFRA_FAILURE_ABORT_RATE, RunAborted, git_sha, is_infra_error
from .schemas import EvalCase

POLICIES = ("single", "cascade", "difficulty")
# Run outcomes after which the cascade asks the strong model. Not `refused`: the
# input screen runs before any model call and would refuse again.
CASCADE_ON = ("escalated", "exhausted", "failed")

RATE_SYSTEM = """You route questions about companies' SEC 10-K filings to one of two \
assistants. Both answer by calling tools that look up reported XBRL figures and \
search the filings' text. The cheaper one is reliable only on easy questions.

easy: asks for one reported figure, for one company and one fiscal year.
hard: anything else: comparing years or companies, growth rates or ratios, \
questions answered by reading and summarising the filing text, or anything ambiguous.

Reply with JSON only: {"difficulty": "easy"} or {"difficulty": "hard"}."""

RATE_SCHEMA: dict[str, Any] = {
    "title": "difficulty",
    "type": "object",
    "properties": {"difficulty": {"type": "string", "enum": ["easy", "hard"]}},
    "required": ["difficulty"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class Arm:
    policy: str
    cheap: str
    strong: str = ""

    @property
    def spec(self) -> str:
        return f"{self.policy}:{self.cheap}" + (f",{self.strong}" if self.strong else "")


def parse_arm(spec: str) -> Arm:
    policy, _, rest = spec.partition(":")
    models = [m.strip() for m in rest.split(",") if m.strip()]
    if policy not in POLICIES or len(models) != (1 if policy == "single" else 2):
        raise ValueError(
            f"arm {spec!r}: want single:<model>, cascade:<cheap>,<strong> "
            "or difficulty:<cheap>,<strong>"
        )
    unpriced = [m for m in models if m not in PRICES_USD_PER_MTOK]
    if unpriced:
        raise ValueError(
            f"no list price for {', '.join(unpriced)}; add it to PRICES_USD_PER_MTOK, with the date"
        )
    return Arm(policy, *models)


class MeteredLLM:
    """The configured provider pinned to one model, logging every call's usage.

    GPT-6 models reject `max_tokens` and any temperature but 1, so they get the
    overrides the fine-tune benchmark ran them with.
    """

    name = "metered"

    def __init__(
        self, model: str, calls: list[dict[str, Any]], inner: LLMProvider | None = None
    ) -> None:
        self.model = model
        self.calls = calls
        self.inner = inner or get_llm()

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        extra = {"extra_body": _OPENAI_REASONING} if self.model.startswith("gpt-6") else {}
        c = self.inner.complete(
            prompt, system=system, model=self.model, temperature=temperature,
            max_tokens=max_tokens, json_schema=json_schema, **extra,
        )
        usage = c.raw.get("usage") or {}
        self.calls.append({
            "model": self.model,
            "served": c.model,
            "prompt": c.prompt_tokens,
            "cached": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
            # Reasoning tokens are billed as output and already inside `completion`.
            "completion": c.completion_tokens,
            "reasoning": int(
                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            ),
            "latency_ms": c.latency_ms,
        })
        return c


def call_cost(call: dict[str, Any]) -> float:
    p_in, p_cached, p_out = PRICES_USD_PER_MTOK[call["model"]]
    fresh = call["prompt"] - call["cached"]
    return (fresh * p_in + call["cached"] * p_cached + call["completion"] * p_out) / 1e6


def rate_difficulty(question: str, llm: LLMProvider) -> str:
    """'easy' or 'hard'; 'unrated' when the reply does not parse (routed as hard)."""
    c = llm.complete(
        question, system=RATE_SYSTEM, temperature=0.0, max_tokens=20, json_schema=RATE_SCHEMA
    )
    try:
        rating = parse_json_strict(c.text).get("difficulty")
    except Exception:
        return "unrated"
    return rating if rating in ("easy", "hard") else "unrated"


@dataclass(slots=True)
class Routed:
    run: AgentRun                   # the run whose answer is graded
    route: list[str]                # the model of each agent run, in order
    outcomes: list[str]             # each agent run's outcome, in the same order
    rating: str = ""                # difficulty policy only
    latency_ms: int = 0             # rating and every agent run, end to end


def answer(arm: Arm, question: str, calls: list[dict[str, Any]]) -> Routed:
    """Route one question. `calls` collects every model call's usage, even on error."""
    started = time.perf_counter()

    def agent(model: str) -> AgentRun:
        return run_agent(question, persist=False, llm=MeteredLLM(model, calls))

    rating = ""
    first = arm.cheap
    if arm.policy == "difficulty":
        rating = rate_difficulty(question, MeteredLLM(arm.cheap, calls))
        first = arm.cheap if rating == "easy" else arm.strong
    runs = [agent(first)]
    route = [first]
    if arm.policy == "cascade" and runs[0].outcome in CASCADE_ON:
        runs.append(agent(arm.strong))
        route.append(arm.strong)
    return Routed(
        run=runs[-1], route=route, outcomes=[r.outcome for r in runs], rating=rating,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def _case_row(arm: Arm, case: EvalCase) -> dict[str, Any]:
    base = {"case_id": case.case_id, "kind": case.kind, "difficulty": case.difficulty}
    calls: list[dict[str, Any]] = []
    try:
        routed = answer(arm, case.question, calls)
        result, record = grade_agent_run(case, routed.run)
    except Exception as exc:
        # The calls made before the error were paid for; keep them in the books.
        return {**base, "error": str(exc)[:500], "calls": calls,
                "cost_usd": round(sum(map(call_cost, calls)), 8)}
    return {
        **base,
        "passed": bool(result.passed),
        "abstained": result.abstained,
        "answer": result.answer[:800],
        "expected": result.expected[:400],
        "why": result.judge_rationale or result.error,
        "route": routed.route,
        "outcomes": routed.outcomes,
        "outcome": routed.run.outcome,
        "rating": routed.rating,
        "confidence": routed.run.confidence,
        "draft_passed": record["draft_passed"],
        "tools": record["tools"],
        "calls": calls,
        "cost_usd": round(sum(map(call_cost, calls)), 8),
        "judge_cost_usd": record["judge_cost_usd"],
        "latency_ms": routed.latency_ms,
        "trace": record["trace"],
    }


def _rows(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_router(
    arm_spec: str, cases: list[EvalCase], label: str, out_dir: str | None = None, progress=None
) -> str:
    """Run every case through one arm, appending a graded row per case. Resumable."""
    arm = parse_arm(arm_spec)
    s = get_settings()
    out_dir = out_dir or os.path.join(s.reports_dir, "router")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{label}.jsonl")
    meta_path = path + ".meta.json"
    case_ids = [c.case_id for c in cases]

    index = index_shape(s.default_strategy)
    if index["chunks"] == 0 or index["embedded"] < index["chunks"]:
        raise RunAborted(
            f"'{s.default_strategy}' has {index['embedded']} of {index['chunks']} chunks "
            "embedded. The agent's search tool would search an empty or partial index."
        )
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta["arm"] != arm.spec or meta["case_ids"] != case_ids:
            raise ValueError(f"{label} was started with another arm or case set; pick a new label")
    else:
        models = [m for m in (arm.cheap, arm.strong) if m]
        meta = {
            "label": label,
            "arm": arm.spec,
            "git_sha": git_sha(),
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "agent": agent_config(),
            "index": index,
            "prices_usd_per_mtok": {m: PRICES_USD_PER_MTOK[m] for m in models},
            "prices_basis": "input / cached input / output, list price read 16 Oct 2026",
            "case_ids": case_ids,
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)

    done = {r["case_id"] for r in _rows(path) if not r.get("error")}
    todo = [c for c in cases if c.case_id not in done]
    infra = 0
    with open(path, "a", encoding="utf-8") as fh:
        for i, case in enumerate(todo, start=1):
            row = _case_row(arm, case)
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            infra += is_infra_error(row.get("error", ""))
            # The same void-run rule as `eval run`: a dead endpoint is not a score.
            if i >= 8 and infra / i > INFRA_FAILURE_ABORT_RATE:
                raise RunAborted(
                    f"{infra} of {i} cases failed before reaching the model "
                    f"({row['error']}). Fix it and re-run the same command to resume."
                )
            if progress:
                progress(i, len(todo), row)
    return path


def sample_cases(cases: list[EvalCase], n: int, seed: int = 11) -> list[EvalCase]:
    """A seeded draw of about n cases, in proportion to each (kind, difficulty), in set order."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[EvalCase]] = {}
    for c in cases:
        strata.setdefault((c.kind, c.difficulty), []).append(c)
    picked: set[str] = set()
    for group in strata.values():
        k = min(len(group), max(1, round(n * len(group) / len(cases))))
        picked.update(c.case_id for c in rng.sample(group, k))
    return [c for c in cases if c.case_id in picked]


# ------------------------------------------------------------------- scoring
def _rate(n: float, d: int) -> float | None:
    return round(n / d, 4) if d else None


def _load(path: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    with open(path + ".meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    rows = _rows(path)
    # A resumed case's newest row wins; superseded rows still count as spend.
    return meta, {r["case_id"]: r for r in rows}, rows


def summarise(meta: dict[str, Any], last: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every reported number for one run, from its rows."""
    final = list(last.values())
    ok = [r for r in final if not r.get("error")]
    num = [r for r in ok if r["kind"] == "numeric"]
    nar = [r for r in ok if r["kind"] == "narrative"]
    correct = sum(r["passed"] for r in num)
    total = sum(r["cost_usd"] for r in ok)
    num_cost = sum(r["cost_usd"] for r in num)
    calls = [c for r in ok for c in r["calls"]]
    lat = [r["latency_ms"] for r in ok]
    strong = parse_arm(meta["arm"]).strong
    by_difficulty = {
        d: f"{sum(r['passed'] for r in num if r['difficulty'] == d)}/"
           f"{sum(r['difficulty'] == d for r in num)}"
        for d in sorted({r["difficulty"] for r in num})
    }
    out: dict[str, Any] = {
        "label": meta["label"],
        "arm": meta["arm"],
        "git_sha": meta["git_sha"],
        "cases": len(final),
        "errors": len(final) - len(ok),
        "numeric": {
            "n": len(num),
            "correct": correct,
            "accuracy": _rate(correct, len(num)),
            "by_difficulty": by_difficulty,
            "wrong": sum(r["outcome"] == "answered" and not r["passed"] and not r["abstained"]
                         for r in num),
            "not_answered": sum(r["outcome"] != "answered" for r in num),
        },
        "narrative_uncalibrated_judge": {"n": len(nar), "passed": sum(r["passed"] for r in nar)},
        "outcomes": dict(Counter(r["outcome"] for r in ok)),
        "strong_share": _rate(sum(strong in r["route"] for r in ok), len(ok)) if strong else None,
        "cost_usd": {
            "total": round(total, 6),
            "per_1k_questions": round(total / len(ok) * 1000, 4) if ok else None,
            "per_1k_correct_numeric": round(num_cost / correct * 1000, 4) if correct else None,
            "judge": round(sum(r["judge_cost_usd"] for r in ok), 6),
            "spent_including_retried_rows": round(sum(r.get("cost_usd", 0) for r in rows), 6),
        },
        "latency_ms": {"p50": _percentile(lat, 0.5), "p95": _percentile(lat, 0.95)},
        "calls_per_question": _rate(len(calls), len(ok)),
        "tokens_per_question": {
            k: round(sum(c[k] for c in calls) / len(ok), 1) if ok else None
            for k in ("prompt", "cached", "completion", "reasoning")
        },
    }
    if meta["arm"].startswith("difficulty"):
        out["ratings"] = dict(Counter(r["rating"] for r in ok))
        # Against the golden set's own labels: hard = anything but a one-figure lookup.
        out["rating_agrees_with_labels"] = _rate(
            sum((r["rating"] == "hard") == (r["kind"] == "narrative" or r["difficulty"] != "single_hop")
                for r in ok),
            len(ok),
        )
    return out


def score(
    paths: list[str], reference: str | None = None, n_boot: int = 2000, seed: int = 7
) -> dict[str, Any]:
    """Every arm's runs, intervals over the numeric cases all runs finished, and
    each arm against `reference`.

    A difference is claimed only when its paired-bootstrap 95% interval excludes
    zero and it is larger, in cases, than the biggest gap between two runs of
    one arm -- the fine-tune benchmark's rule.
    """
    runs = [_load(p) for p in paths]
    ids = sorted(set.intersection(*(
        {cid for cid, r in last.items() if r["kind"] == "numeric" and not r.get("error")}
        for _, last, _ in runs
    )))
    if not ids:
        raise ValueError("no numeric case was finished by every run")
    arms: dict[str, list[dict[str, dict[str, Any]]]] = {}
    for meta, last, _ in runs:
        arms.setdefault(meta["arm"], []).append(last)
    if reference and reference not in arms:
        raise ValueError(f"reference {reference!r} is not among the runs' arms: {', '.join(arms)}")

    rng = random.Random(seed)
    draws = [[rng.randrange(len(ids)) for _ in ids] for _ in range(n_boot)]

    def per_case(group: list[dict[str, dict[str, Any]]]) -> list[float]:
        return [statistics.fmean(bool(last[cid]["passed"]) for last in group) for cid in ids]

    def interval(values: list[float]) -> list[float]:
        means = sorted(statistics.fmean(values[i] for i in d) for d in draws)
        return [round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot) - 1], 4)]

    out: dict[str, Any] = {"numeric_cases_common": len(ids), "arms": {}}
    for spec, group in arms.items():
        pc = per_case(group)
        correct = [sum(bool(last[cid]["passed"]) for cid in ids) for last in group]
        common: dict[str, Any] = {
            "correct_by_run": correct,
            "accuracy_mean": round(statistics.fmean(pc), 4),
            "ci95": interval(pc),
            "spread_cases": max(correct) - min(correct),
        }
        if len(group) >= 2:
            a, b = group[0], group[1]
            common["flips_run1_to_run2"] = {
                "fail_to_pass": sum(not a[c]["passed"] and b[c]["passed"] for c in ids),
                "pass_to_fail": sum(a[c]["passed"] and not b[c]["passed"] for c in ids),
            }
        out["arms"][spec] = {
            "runs": [summarise(m, last, rows) for m, last, rows in runs if m["arm"] == spec],
            "numeric_common": common,
        }

    spreads = [a["numeric_common"]["spread_cases"] for a in out["arms"].values() if len(a["runs"]) > 1]
    noise = max(spreads) if spreads else None
    out["largest_same_arm_spread_cases"] = noise
    if reference:
        ref = per_case(arms[reference])
        for spec, group in arms.items():
            if spec == reference:
                continue
            diff = [x - y for x, y in zip(per_case(group), ref, strict=True)]
            d = statistics.fmean(diff)
            lo, hi = interval(diff)
            out["arms"][spec]["vs_reference"] = {
                "reference": reference,
                "diff": round(d, 4),
                "diff_cases": round(d * len(ids), 1),
                "ci95": [lo, hi],
                "claimed": (lo > 0 or hi < 0) and noise is not None and abs(d) * len(ids) > noise,
            }
    return out
