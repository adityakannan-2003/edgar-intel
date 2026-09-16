"""Controlled single-case probe: what actually reaches the answering model.

Aggregate metrics say a run scored 0.62. They do not say *why one case failed*,
and on `nar-AAPL-legal` the aggregate was actively misleading: all five relevant
chunks were inside the hybrid top-50, at pre-rerank ranks 9, 11, 13, 21 and 29,
and `top_n=8` cut every one of them before the model saw anything. The answer
was graded as a quality failure. Nothing about it was a quality failure.

This module runs one case across a grid of `top_n` values with reranking on and
off, and records, per arm:

  * whether the judge passed it
  * how many relevant chunks reached the final context, and at what ranks
  * where those chunks sat *before* truncation, which is what distinguishes
    "retrieval never found it" from "ranking found it and threw it away"
  * retrieve / rerank / total latency
  * prompt and completion tokens, and dollar cost
  * the git SHA, top_n and rerank flag of the arm that produced them

Two rules this file exists to enforce:

**One variable per arm.** Nothing but `top_n` and `use_rerank` changes between
arms. Grader, prompt, strategy, k, embeddings and chunking are held fixed, and
the config each arm ran under is written into the report so a later reader does
not have to trust that claim.

**The probe calls the production path.** It goes through `answer_question`, the
same function `run_suite` calls, rather than re-assembling retrieval beside it.
A probe that reimplements the pipeline eventually measures a pipeline nobody
ships.

Results are written to a report file and **not** to `eval_runs`. These are
experiments, not baselines; mixing them into the run table is how a metrics
history stops meaning anything.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import get_settings
from ..retrieval.search import infer_expected_items, search
from .judge import grade
from .runner import answer_question, git_sha
from .schemas import EvalCase

DEFAULT_TOP_N_GRID = (8, 12, 16, 20)


@dataclass(slots=True)
class ProbeArm:
    """One (top_n, use_rerank) measurement of one case."""

    case_id: str
    top_n: int
    use_rerank: bool
    git_sha: str
    strategy: str
    mode: str
    k: int
    item_boost: float

    passed: bool = False
    judge_rationale: str = ""
    answer: str = ""

    n_relevant: int = 0
    relevant_in_context: int = 0
    relevant_final_ranks: list[int] = field(default_factory=list)
    relevant_pre_truncation_ranks: list[int] = field(default_factory=list)
    context_items: list[str] = field(default_factory=list)
    inferred_items: list[str] = field(default_factory=list)

    retrieve_ms: int = 0
    rerank_ms: int = 0
    total_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: str = ""

    def row(self) -> dict[str, Any]:
        return {
            "top_n": self.top_n,
            "rerank": self.use_rerank,
            "pass": self.passed,
            "relevant_in_context": f"{self.relevant_in_context}/{self.n_relevant}",
            "final_ranks": self.relevant_final_ranks,
            "pre_trunc_ranks": self.relevant_pre_truncation_ranks,
            "retrieve_ms": self.retrieve_ms,
            "rerank_ms": self.rerank_ms,
            "total_ms": self.total_ms,
            "p_tok": self.prompt_tokens,
            "c_tok": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


def candidate_ranks(
    case: EvalCase,
    strategy: str,
    mode: str,
    use_rerank: bool,
    k: int,
    item_boost: float,
) -> list[int]:
    """Where the relevant chunks sit before `top_n` truncation.

    Run with `top_n=k`, so nothing is cut. The gap between these ranks and the
    final ones is the loss attributable to truncation alone -- the number the
    whole exercise is trying to isolate.
    """
    result = search(
        case.question,
        strategy=strategy,
        mode=mode,
        use_rerank=use_rerank,
        k=k,
        top_n=k,
        ticker=case.ticker,
        fiscal_year=case.fiscal_year,
        item_boost=item_boost,
    )
    relevant = set(case.relevant_chunk_ids)
    return [i for i, cid in enumerate(result.ids, start=1) if cid in relevant]


def run_arm(
    case: EvalCase,
    top_n: int,
    use_rerank: bool,
    strategy: str,
    mode: str,
    k: int,
    item_boost: float,
    sha: str,
) -> ProbeArm:
    arm = ProbeArm(
        case_id=case.case_id,
        top_n=top_n,
        use_rerank=use_rerank,
        git_sha=sha,
        strategy=strategy,
        mode=mode,
        k=k,
        item_boost=item_boost,
        n_relevant=len(case.relevant_chunk_ids),
        inferred_items=infer_expected_items(case.question),
    )

    started = time.perf_counter()
    try:
        arm.relevant_pre_truncation_ranks = candidate_ranks(
            case, strategy, mode, use_rerank, k, item_boost
        )
        answer, retrieval, latency, p_tok, c_tok, result = answer_question(
            case, strategy, mode, use_rerank, k, top_n
        )
    except Exception as exc:  # noqa: BLE001 - the message is the finding
        arm.error = str(exc)[:600]
        arm.total_ms = int((time.perf_counter() - started) * 1000)
        return arm

    relevant = set(case.relevant_chunk_ids)
    arm.answer = answer
    arm.relevant_final_ranks = [
        i for i, hit in enumerate(result.hits, start=1) if hit.chunk_id in relevant
    ]
    arm.relevant_in_context = len(arm.relevant_final_ranks)
    arm.context_items = [hit.item or "?" for hit in result.hits]
    arm.retrieve_ms = int(retrieval.get("retrieve_ms", 0))
    arm.rerank_ms = int(retrieval.get("rerank_ms", 0))
    arm.total_ms = latency
    arm.prompt_tokens = p_tok
    arm.completion_tokens = c_tok

    passed, _score, rationale, verdict = grade(case, answer)
    arm.passed = passed
    arm.judge_rationale = rationale[:400]
    if verdict:
        arm.prompt_tokens += verdict.prompt_tokens
        arm.completion_tokens += verdict.completion_tokens
    arm.cost_usd = get_settings().cost_usd(arm.prompt_tokens, arm.completion_tokens)
    return arm


def probe(
    cases: list[EvalCase],
    top_n_grid: tuple[int, ...] = DEFAULT_TOP_N_GRID,
    rerank_settings: tuple[bool, ...] = (True, False),
    repeats: int = 1,
    strategy: str | None = None,
    mode: str = "hybrid",
    k: int | None = None,
    item_boost: float | None = None,
    out_path: str = "reports/context_probe.json",
    progress=None,
) -> dict[str, Any]:
    """Every (top_n, rerank) arm over every case, repeated `repeats` times.

    `repeats` exists because "passes reliably" is a claim about variance, and a
    single pass at `top_n=16` does not establish it. The generation temperature
    is 0.0, so repeats measure the residual non-determinism of the provider
    rather than sampling noise -- which is exactly the thing that decides whether
    a threshold is safe to ship.
    """
    s = get_settings()
    strategy = strategy or s.default_strategy
    k = k or s.retrieve_k
    boost = s.item_boost_weight if item_boost is None else item_boost
    sha = git_sha()

    arms: list[ProbeArm] = []
    for case in cases:
        for use_rerank in rerank_settings:
            for top_n in top_n_grid:
                for _ in range(repeats):
                    arm = run_arm(case, top_n, use_rerank, strategy, mode, k, boost, sha)
                    arms.append(arm)
                    if progress:
                        progress(arm)

    payload = {
        "git_sha": sha,
        "config": {
            "strategy": strategy,
            "mode": mode,
            "k": k,
            "item_boost_weight": boost,
            "llm_model": s.llm_model,
            "judge_model": s.judge_model,
            "embed_model": s.embed_model,
            "rerank_model": s.rerank_model,
            "repeats": repeats,
        },
        "cases": [c.case_id for c in cases],
        "arms": [asdict(a) for a in arms],
        "rows": [a.row() for a in arms],
        "recommendation": recommend(arms),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def recommend(arms: list[ProbeArm]) -> str:
    """The smallest top_n that passes every repeat, and whether that is the real fix."""
    if not arms:
        return "no arms run"

    by_setting: dict[tuple[int, bool], list[ProbeArm]] = {}
    for a in arms:
        by_setting.setdefault((a.top_n, a.use_rerank), []).append(a)

    reliable = sorted(
        key for key, group in by_setting.items() if group and all(g.passed for g in group)
    )
    lines: list[str] = []
    if not reliable:
        lines.append(
            "No top_n in the grid passes reliably. Truncation is not the only "
            "problem -- widen the grid, then look at ranking."
        )
    else:
        top_n, use_rerank = reliable[0]
        lines.append(
            f"Smallest reliable setting: top_n={top_n}, rerank={use_rerank}."
        )

    # Was the evidence ever in the candidate set at all? If yes, this is a
    # ranking/truncation problem and no amount of re-embedding will help.
    ever_found = max((len(a.relevant_pre_truncation_ranks) for a in arms), default=0)
    best_final = max((a.relevant_in_context for a in arms), default=0)
    if ever_found and best_final < ever_found:
        worst = min(
            (a for a in arms if a.relevant_pre_truncation_ranks),
            key=lambda a: min(a.relevant_pre_truncation_ranks),
        )
        lines.append(
            f"Evidence IS in the candidate set (best pre-truncation ranks "
            f"{worst.relevant_pre_truncation_ranks}) but only {best_final} of "
            f"{ever_found} reach the model at the grid's widest setting. The "
            "bottleneck is ranking and context selection, not candidate generation."
        )

    deep = [a for a in arms if a.relevant_pre_truncation_ranks
            and min(a.relevant_pre_truncation_ranks) > 5]
    if deep:
        lines.append(
            "The first relevant chunk never reaches the top 5, so raising top_n "
            "buys the answer by paying for more context rather than by ranking "
            "better. Treat a top_n increase as mitigation and keep the ranking "
            "work open."
        )
    return " ".join(lines)
