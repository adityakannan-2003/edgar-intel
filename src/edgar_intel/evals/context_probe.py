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
from ..retrieval.search import DEFAULT_CONTEXT_MAX_CHARS, infer_expected_items
from .evidence import labels_for
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

    context_max_chars: int = DEFAULT_CONTEXT_MAX_CHARS
    context_packing: str = "greedy-stop"

    n_relevant: int = 0
    inferred_items: list[str] = field(default_factory=list)

    # --- stage 1: what the ranker produced, uncut
    candidate_ids: list[str] = field(default_factory=list)
    relevant_pre_truncation_ranks: list[int] = field(default_factory=list)

    # --- stage 2: what survived top_n
    selected_ids: list[str] = field(default_factory=list)
    relevant_final_ranks: list[int] = field(default_factory=list)
    relevant_in_context: int = 0
    context_items: list[str] = field(default_factory=list)

    # --- stage 3: what build_context actually put in the prompt
    included_ids: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)
    n_included: int = 0
    context_chars_before_cap: int = 0
    context_chars_after_cap: int = 0
    cap_binding: bool = False
    first_excluded_id: str | None = None
    first_excluded_rank: int | None = None
    first_excluded_chars: int = 0
    would_fit_with_skip: int = 0
    relevant_in_prompt: int = 0
    relevant_ids_in_prompt: list[str] = field(default_factory=list)
    relevant_ids_lost_to_cap: list[str] = field(default_factory=list)

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
            "boost": self.item_boost,
            "pass": self.passed,
            "pre_trunc_ranks": self.relevant_pre_truncation_ranks,
            "after_top_n": f"{self.relevant_in_context}/{self.n_relevant}",
            "in_prompt": f"{self.relevant_in_prompt}/{self.n_relevant}",
            "hits_incl": f"{self.n_included}/{len(self.selected_ids)}",
            "ctx_chars": f"{self.context_chars_after_cap}/{self.context_chars_before_cap}",
            "cap": "BINDING" if self.cap_binding else "-",
            "total_ms": self.total_ms,
            "p_tok": self.prompt_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


def run_arm(
    case: EvalCase,
    top_n: int,
    use_rerank: bool,
    strategy: str,
    mode: str,
    k: int,
    item_boost: float,
    sha: str,
    context_max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    context_packing: str = "greedy-stop",
) -> ProbeArm:
    """One arm, one retrieval call.

    The previous version made two: a diagnostic `search()` for the
    pre-truncation ranks and `answer_question()` for the answer. They were
    configured separately, `item_boost` reached only the first, and the probe
    reported boosted ranks beside an answer generated from unboosted retrieval.
    Both views now come from the same `RetrievalResult`, so they cannot
    disagree.
    """
    arm = ProbeArm(
        case_id=case.case_id,
        top_n=top_n,
        use_rerank=use_rerank,
        git_sha=sha,
        strategy=strategy,
        mode=mode,
        k=k,
        item_boost=item_boost,
        context_max_chars=context_max_chars,
        context_packing=context_packing,
        n_relevant=len(case.relevant_chunk_ids),
        inferred_items=infer_expected_items(case.question),
    )

    started = time.perf_counter()
    try:
        answer, retrieval, latency, p_tok, c_tok, result, ctx = answer_question(
            case,
            strategy,
            mode,
            use_rerank,
            k,
            top_n,
            item_boost=item_boost,
            context_max_chars=context_max_chars,
            context_packing=context_packing,
        )
    except Exception as exc:  # noqa: BLE001 - the message is the finding
        arm.error = str(exc)[:600]
        arm.total_ms = int((time.perf_counter() - started) * 1000)
        return arm

    relevant = set(case.relevant_chunk_ids)
    arm.answer = answer

    # Stage 1 -- the uncut ordering, from the same call that produced the answer.
    arm.candidate_ids = result.candidate_ids[:50]
    arm.relevant_pre_truncation_ranks = result.ranks_of(relevant, pre_truncation=True)

    # Stage 2 -- after top_n.
    arm.selected_ids = result.ids
    arm.relevant_final_ranks = result.ranks_of(relevant)
    arm.relevant_in_context = len(arm.relevant_final_ranks)
    arm.context_items = [hit.item or "?" for hit in result.hits]

    # Stage 3 -- what build_context actually put in front of the model.
    arm.included_ids = ctx.included_ids
    arm.dropped_ids = ctx.dropped_ids
    arm.n_included = ctx.n_included
    arm.context_chars_before_cap = ctx.chars_before_cap
    arm.context_chars_after_cap = ctx.chars_after_cap
    arm.cap_binding = ctx.cap_binding
    arm.first_excluded_id = ctx.first_excluded_id
    arm.first_excluded_rank = ctx.first_excluded_rank
    arm.first_excluded_chars = ctx.first_excluded_chars
    arm.would_fit_with_skip = ctx.would_fit_with_skip
    arm.relevant_ids_in_prompt = [cid for cid in ctx.included_ids if cid in relevant]
    arm.relevant_in_prompt = len(arm.relevant_ids_in_prompt)
    arm.relevant_ids_lost_to_cap = [cid for cid in ctx.dropped_ids if cid in relevant]

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


def default_report_name(
    context_max_chars: int,
    item_boost: float,
    use_rerank_settings: tuple[bool, ...],
    packing: str,
    prefix: str = "reports/context_probe",
) -> str:
    """A filename that encodes the arm, so two runs cannot collide.

    The 20k and 24k runs were both written to `context_probe_16k.json` because
    the name was typed by hand and `--out` carried over. The 20k evidence is
    gone. A harness that silently overwrites the previous experiment is the same
    defect as a report that silently truncates failures: the measurement is
    fine, the record of it is not.
    """
    parts = [f"{context_max_chars // 1000}k"]
    parts.append(f"boost{item_boost:g}".replace(".", "_"))
    if use_rerank_settings == (True,):
        parts.append("rerank")
    elif use_rerank_settings == (False,):
        parts.append("norerank")
    if packing != "greedy-stop":
        parts.append(packing)
    return f"{prefix}_{'_'.join(parts)}.json"


def probe(
    cases: list[EvalCase],
    top_n_grid: tuple[int, ...] = DEFAULT_TOP_N_GRID,
    rerank_settings: tuple[bool, ...] = (True, False),
    repeats: int = 1,
    strategy: str | None = None,
    mode: str = "hybrid",
    k: int | None = None,
    item_boost: float | None = None,
    context_max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    context_packing: str = "greedy-stop",
    out_path: str | None = None,
    overwrite: bool = False,
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
    # Gold ids from another index would report every relevant chunk as lost.
    cases, evidence_labels = labels_for(cases, strategy)

    out_path = out_path or default_report_name(
        context_max_chars, boost, rerank_settings, context_packing
    )
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Experiment records are evidence; pass "
            "--overwrite to replace it, or let the default name encode the arm."
        )

    arms: list[ProbeArm] = []
    for case in cases:
        for use_rerank in rerank_settings:
            for top_n in top_n_grid:
                for _ in range(repeats):
                    arm = run_arm(
                        case, top_n, use_rerank, strategy, mode, k, boost, sha,
                        context_max_chars=context_max_chars,
                        context_packing=context_packing,
                    )
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
            "context_max_chars": context_max_chars,
            "context_packing": context_packing,
            "llm_model": s.llm_model,
            "judge_model": s.judge_model,
            "embed_model": s.embed_model,
            "rerank_model": s.rerank_model,
            "repeats": repeats,
            "evidence_labels": evidence_labels,
        },
        "out_path": out_path,
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
    # ranking/selection problem and no amount of re-embedding will help.
    ever_found = max((len(a.relevant_pre_truncation_ranks) for a in arms), default=0)
    best_in_prompt = max((a.relevant_in_prompt for a in arms), default=0)
    if ever_found and best_in_prompt < ever_found:
        best_ranked = min(
            (a for a in arms if a.relevant_pre_truncation_ranks),
            key=lambda a: min(a.relevant_pre_truncation_ranks),
        )
        lines.append(
            f"Evidence IS in the candidate set (best pre-truncation ranks "
            f"{best_ranked.relevant_pre_truncation_ranks}) but only "
            f"{best_in_prompt} of {ever_found} reach the prompt at the grid's "
            "widest setting. The bottleneck is ranking and context selection, "
            "not candidate generation."
        )

    # Distinguish the two ways a passage disappears. They have different fixes
    # and, before this was instrumented, looked identical from the outside.
    lost_to_cap = [a for a in arms if a.relevant_ids_lost_to_cap]
    if lost_to_cap:
        worst = max(lost_to_cap, key=lambda a: len(a.relevant_ids_lost_to_cap))
        lines.append(
            f"RELEVANT EVIDENCE IS BEING DISCARDED BY THE CHARACTER CAP, not by "
            f"top_n: {len(worst.relevant_ids_lost_to_cap)} relevant chunk(s) "
            f"survived ranking, entered the top_n={worst.top_n} selection, and "
            f"were then cut by max_chars={worst.context_max_chars}. Raising "
            "top_n cannot fix this."
        )

    capped = [a for a in arms if a.cap_binding]
    if capped:
        widest = max(capped, key=lambda a: a.top_n)
        flat = {a.n_included for a in capped}
        lines.append(
            f"The {widest.context_max_chars}-char cap binds: at top_n="
            f"{widest.top_n} only {widest.n_included} of {len(widest.selected_ids)} "
            f"selected passages reach the prompt "
            f"({widest.context_chars_after_cap} of "
            f"{widest.context_chars_before_cap} chars). "
            + (
                f"Included count is {flat.pop()} across every capped arm, so "
                "raising top_n changes nothing downstream. "
                if len(flat) == 1
                else ""
            )
            + (
                f"A skip-oversized packing policy would have admitted "
                f"{widest.would_fit_with_skip} instead of {widest.n_included}."
                if widest.would_fit_with_skip > widest.n_included
                else ""
            )
        )

    deep = [a for a in arms if a.relevant_pre_truncation_ranks
            and min(a.relevant_pre_truncation_ranks) > 5]
    if deep and len(deep) == len([a for a in arms if a.relevant_pre_truncation_ranks]):
        lines.append(
            "The first relevant chunk never reaches the top 5 in any arm, so "
            "raising top_n buys the answer by paying for more context rather "
            "than by ranking better. Treat a top_n increase as mitigation and "
            "keep the ranking work open."
        )
    return " ".join(lines)
