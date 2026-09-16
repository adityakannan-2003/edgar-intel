"""Judge calibration on frozen pairs: one variable per comparison.

Retrieval and generation are held out entirely. A frozen pair is a saved
(question, reference, source context, answer) record, so every replay grades
identical bytes and the only things that move are the ones under test:

    contract   v1 (shipped) vs v2 (rewritten rubric)
    model      gpt-4o-mini vs anything else

Changing both at once proves nothing -- if v1+mini fails and v2+gpt-4o passes,
the experiment has not said which caused it. `single_variable_comparisons()`
enumerates the pairs of cells that differ in exactly one dimension, and the
report prints only those as conclusions.

**Negative controls are the point of this module.** Replaying one known-good
answer until it passes does not measure a judge, it measures whether the rubric
is permissive. A contract that passes everything scores perfectly on that test
and is worthless -- it is the degenerate case `cohens_kappa` already has a test
for. So each frozen good answer is mutated into answers that *should* fail, one
per rubric dimension:

    contradiction   a figure changed so it contradicts the reference
    omission        material facts the reference requires, removed
    hedge           a generic non-answer
    fabrication     a proceeding that appears in neither reference nor context

A contract is only better than v1 if it passes the good answer **and** fails
these. Both numbers are reported; neither alone is a result.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import get_settings
from .judge import judge_narrative
from .judge_contracts import CONTRACTS
from .schemas import EvalCase


@dataclass(slots=True)
class FrozenPair:
    """One graded example, detached from the pipeline that produced it."""

    pair_id: str
    question: str
    reference: str
    answer: str
    source_context: str = ""
    expected_verdict: bool = True
    label: str = "good"          # good | contradiction | omission | hedge | fabrication
    provenance: str = ""

    def as_case(self) -> EvalCase:
        return EvalCase(
            case_id=self.pair_id,
            kind="narrative",
            question=self.question,
            expected=self.reference,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------- mutations
_MONEY = re.compile(r"([€$£])\s?(\d[\d,.]*)\s?(million|billion|thousand)?", re.I)


def mutate_contradiction(pair: FrozenPair) -> FrozenPair | None:
    """Change a figure so the answer contradicts the reference.

    Rubric dimension 1. A judge that still passes this is not checking facts at
    all, and the rewrite has traded one failure mode for a worse one.
    """
    match = _MONEY.search(pair.answer)
    if not match:
        return None
    symbol, digits, scale = match.groups()
    original = int(re.sub(r"[^\d]", "", digits) or 0)
    if not original:
        return None
    altered = f"{symbol}{original * 3:,}" + (f" {scale}" if scale else "")
    return FrozenPair(
        pair_id=f"{pair.pair_id}::contradiction",
        question=pair.question,
        reference=pair.reference,
        answer=pair.answer[: match.start()] + altered + pair.answer[match.end():],
        source_context=pair.source_context,
        expected_verdict=False,
        label="contradiction",
        provenance=f"mutated from {pair.pair_id}: {match.group(0)} -> {altered}",
    )


def mutate_omission(pair: FrozenPair, drop_from_sentence: int = 2) -> FrozenPair | None:
    """Keep only the opening claims, dropping material coverage the reference requires.

    Rubric dimension 2.
    """
    sentences = re.split(r"(?<=[.;])\s+", pair.answer.strip())
    if len(sentences) <= drop_from_sentence:
        return None
    kept = " ".join(sentences[:drop_from_sentence])
    return FrozenPair(
        pair_id=f"{pair.pair_id}::omission",
        question=pair.question,
        reference=pair.reference,
        answer=kept,
        source_context=pair.source_context,
        expected_verdict=False,
        label="omission",
        provenance=f"mutated from {pair.pair_id}: kept {drop_from_sentence} of {len(sentences)} sentences",
    )


HEDGE_ANSWER = (
    "The company discloses various legal proceedings and claims arising in the "
    "ordinary course of business. The outcomes are uncertain and could have a "
    "material effect on the company's financial condition or results of operations."
)


def mutate_hedge(pair: FrozenPair) -> FrozenPair:
    """A generically true non-answer. Rubric: "a hedge that avoids answering is not correct"."""
    return FrozenPair(
        pair_id=f"{pair.pair_id}::hedge",
        question=pair.question,
        reference=pair.reference,
        answer=HEDGE_ANSWER,
        source_context=pair.source_context,
        expected_verdict=False,
        label="hedge",
        provenance=f"generic hedge substituted for {pair.pair_id}",
    )


FABRICATION = (
    " The company also discloses a $4.2 billion securities class action in the "
    "Southern District of New York brought by pension funds over restated revenue."
)


def mutate_fabrication(pair: FrozenPair) -> FrozenPair:
    """Correct-looking detail supported by neither reference nor context.

    Rubric dimension 3. This is the control that keeps "extra detail is allowed"
    from becoming "anything goes" -- the judge has the source context and can
    check, which is precisely what v1 could not do.
    """
    return FrozenPair(
        pair_id=f"{pair.pair_id}::fabrication",
        question=pair.question,
        reference=pair.reference,
        answer=pair.answer.rstrip() + FABRICATION,
        source_context=pair.source_context,
        expected_verdict=False,
        label="fabrication",
        provenance=f"fabricated proceeding appended to {pair.pair_id}",
    )


def build_controls(pair: FrozenPair) -> list[FrozenPair]:
    """The good answer plus every negative control derivable from it."""
    out = [pair]
    for fn in (mutate_contradiction, mutate_omission):
        mutated = fn(pair)
        if mutated:
            out.append(mutated)
    out.append(mutate_hedge(pair))
    out.append(mutate_fabrication(pair))
    return out


# --------------------------------------------------------------------- grid
@dataclass(slots=True)
class JudgeRun:
    pair_id: str
    label: str
    contract: str
    model: str
    repeat: int
    expected_verdict: bool
    verdict: bool | None = None
    correct: bool = False
    rationale: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


@dataclass(slots=True)
class CellSummary:
    contract: str
    model: str
    n_good: int = 0
    good_passed: int = 0
    n_bad: int = 0
    bad_failed: int = 0
    errors: int = 0
    per_label: dict[str, str] = field(default_factory=dict)

    @property
    def good_pass_rate(self) -> float:
        return self.good_passed / self.n_good if self.n_good else 0.0

    @property
    def bad_catch_rate(self) -> float:
        return self.bad_failed / self.n_bad if self.n_bad else 0.0

    @property
    def balanced_accuracy(self) -> float:
        """The single number, and it needs both halves.

        A permissive judge scores 1.0 on good and 0.0 on bad. A strict one does
        the reverse. Only a judge that does both well clears 0.9 here.
        """
        if not self.n_good or not self.n_bad:
            return 0.0
        return (self.good_pass_rate + self.bad_catch_rate) / 2

    def row(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "model": self.model,
            "good_pass": f"{self.good_passed}/{self.n_good}",
            "bad_caught": f"{self.bad_failed}/{self.n_bad}",
            "balanced_acc": round(self.balanced_accuracy, 3),
            "errors": self.errors,
            **self.per_label,
        }


def run_grid(
    pairs: list[FrozenPair],
    contracts: tuple[str, ...] = ("v1", "v2"),
    models: tuple[str, ...] = (),
    repeats: int = 5,
    progress=None,
) -> list[JudgeRun]:
    s = get_settings()
    models = models or (s.judge_model,)
    runs: list[JudgeRun] = []
    for contract in contracts:
        if contract not in CONTRACTS:
            raise ValueError(f"unknown contract {contract!r}")
        for model in models:
            for pair in pairs:
                for i in range(repeats):
                    run = JudgeRun(
                        pair_id=pair.pair_id,
                        label=pair.label,
                        contract=contract,
                        model=model,
                        repeat=i + 1,
                        expected_verdict=pair.expected_verdict,
                    )
                    try:
                        verdict = judge_narrative(
                            pair.as_case(),
                            pair.answer,
                            contract=contract,
                            source_context=pair.source_context,
                            model=model,
                        )
                        run.verdict = verdict.verdict
                        run.correct = verdict.verdict == pair.expected_verdict
                        run.rationale = verdict.rationale
                        run.latency_ms = verdict.latency_ms
                        run.prompt_tokens = verdict.prompt_tokens
                        run.completion_tokens = verdict.completion_tokens
                    except Exception as exc:  # noqa: BLE001
                        run.error = str(exc)[:400]
                    runs.append(run)
                    if progress:
                        progress(run)
    return runs


def summarise(runs: list[JudgeRun]) -> list[CellSummary]:
    cells: dict[tuple[str, str], CellSummary] = {}
    label_totals: dict[tuple[str, str, str], list[int]] = {}
    for r in runs:
        cell = cells.setdefault((r.contract, r.model), CellSummary(r.contract, r.model))
        if r.error:
            cell.errors += 1
            continue
        if r.expected_verdict:
            cell.n_good += 1
            cell.good_passed += int(bool(r.verdict))
        else:
            cell.n_bad += 1
            cell.bad_failed += int(not r.verdict)
        bucket = label_totals.setdefault((r.contract, r.model, r.label), [0, 0])
        bucket[1] += 1
        bucket[0] += int(r.correct)

    for (contract, model, label), (correct, total) in label_totals.items():
        cells[(contract, model)].per_label[label] = f"{correct}/{total}"
    return sorted(cells.values(), key=lambda c: (c.contract, c.model))


def single_variable_comparisons(cells: list[CellSummary]) -> list[dict[str, Any]]:
    """Only the cell pairs that differ in exactly one dimension.

    Any other comparison confounds the rubric with the model, which is the
    mistake this whole module exists to prevent.
    """
    out: list[dict[str, Any]] = []
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            same_model = a.model == b.model
            same_contract = a.contract == b.contract
            if same_model == same_contract:
                continue  # both differ, or neither -- not a clean comparison
            dimension = "contract" if same_model else "model"
            out.append(
                {
                    "varies": dimension,
                    "from": f"{a.contract}/{a.model}",
                    "to": f"{b.contract}/{b.model}",
                    "good_pass": f"{a.good_pass_rate:.2f} -> {b.good_pass_rate:.2f}",
                    "bad_caught": f"{a.bad_catch_rate:.2f} -> {b.bad_catch_rate:.2f}",
                    "balanced_acc": f"{a.balanced_accuracy:.3f} -> {b.balanced_accuracy:.3f}",
                    "delta": round(b.balanced_accuracy - a.balanced_accuracy, 3),
                }
            )
    return out


def verdict(cells: list[CellSummary], comparisons: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    if not cells:
        return "no cells run"

    best = max(cells, key=lambda c: c.balanced_accuracy)
    lines.append(
        f"Best cell: contract={best.contract} model={best.model}, "
        f"balanced accuracy {best.balanced_accuracy:.3f} "
        f"(good {best.good_passed}/{best.n_good}, bad caught {best.bad_failed}/{best.n_bad})."
    )

    permissive = [c for c in cells if c.n_bad and c.bad_catch_rate < 0.6]
    if permissive:
        names = ", ".join(f"{c.contract}/{c.model}" for c in permissive)
        lines.append(
            f"PERMISSIVE: {names} let through more than 40% of answers that should "
            "fail. A rubric that passes the good answer by passing everything is a "
            "worse instrument than the strict one it replaced, not a better one."
        )

    strict = [c for c in cells if c.n_good and c.good_pass_rate < 0.6]
    if strict:
        names = ", ".join(f"{c.contract}/{c.model}" for c in strict)
        lines.append(f"STRICT: {names} failed a known-good answer more often than not.")

    contract_moves = [c for c in comparisons if c["varies"] == "contract"]
    model_moves = [c for c in comparisons if c["varies"] == "model"]
    if contract_moves:
        biggest = max(contract_moves, key=lambda c: abs(c["delta"]))
        lines.append(
            f"Prompt effect (model held fixed): {biggest['from']} -> {biggest['to']} "
            f"moves balanced accuracy {biggest['balanced_acc']}."
        )
    if model_moves:
        biggest = max(model_moves, key=lambda c: abs(c["delta"]))
        lines.append(
            f"Model effect (contract held fixed): {biggest['from']} -> {biggest['to']} "
            f"moves balanced accuracy {biggest['balanced_acc']}."
        )
    if contract_moves and model_moves:
        cm = max(abs(c["delta"]) for c in contract_moves)
        mm = max(abs(c["delta"]) for c in model_moves)
        lines.append(
            "The prompt is the dominant factor."
            if cm > mm
            else "The model is the dominant factor -- a rubric rewrite will not be enough."
        )
    return " ".join(lines)


# ------------------------------------------------------------------ storage
def save_pairs(pairs: list[FrozenPair], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([p.as_dict() for p in pairs], fh, indent=2)
    return path


def load_pairs(path: str) -> list[FrozenPair]:
    with open(path, encoding="utf-8") as fh:
        return [FrozenPair(**row) for row in json.load(fh)]


def freeze_from_autopsy(report_path: str, pair_id: str = "") -> FrozenPair:
    """Build a frozen pair from a saved autopsy, with its source context."""
    with open(report_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return FrozenPair(
        pair_id=pair_id or data["case_id"],
        question=data["question"],
        reference=data["reference"],
        answer=data["answer"],
        source_context=data.get("context_text", ""),
        expected_verdict=True,
        label="good",
        provenance=f"{report_path} @ {data.get('git_sha', '?')}",
    )


def save_runs(runs: list[JudgeRun], cells: list[CellSummary], out_path: str) -> dict[str, Any]:
    comparisons = single_variable_comparisons(cells)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cells": [c.row() for c in cells],
        "single_variable_comparisons": comparisons,
        "verdict": verdict(cells, comparisons),
        "runs": [asdict(r) for r in runs],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload
