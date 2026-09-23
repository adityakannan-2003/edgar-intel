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

# Repeats per pair for a calibration comparison.
#
# Nine, not three, because three was not enough to distinguish a rubric change
# from the judge disagreeing with itself. Two nominally identical context-free
# runs of v2 on the same 24 labels produced kappa 0.4167 (FP 7, FN 0) and 0.5833
# (FP 4, FN 1). That spread is the same size as every contract difference
# measured so far, which means three repeats could not resolve the question being
# asked of them.
#
# Raising repeats makes the per-pair majority harder to flip; it does not widen
# n, which is the label count. Both matter and they are different knobs.
CALIBRATION_REPEATS = 9


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
    # The experimental condition this verdict belongs to. `structure` is the
    # prompt layout; `replicate` distinguishes two executions of an otherwise
    # identical cell, which is how run-to-run variance gets measured instead of
    # discovered by accident.
    structure: str = "current"
    replicate: int = 1
    context_supplied: bool = False
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
    # Confusion against the human label, which is what kappa is computed from.
    # human PASS/judge PASS, human PASS/judge FAIL, human FAIL/judge PASS,
    # human FAIL/judge FAIL.
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0

    @property
    def kappa(self) -> float:
        """Cohen's kappa against the human labels.

        Reported instead of raw agreement because agreement flatters a skewed
        judge. v2 agreed 17/24 = 70.8% with the human and scored kappa 0.4167:
        it passed 79% of answers against a human pass rate of 50%, so chance
        agreement was already 0.5 and the raw number was mostly luck.
        """
        n = self.tp + self.fn + self.fp + self.tn
        if not n:
            return 0.0
        p_o = (self.tp + self.tn) / n
        p_e = ((self.tp + self.fn) / n) * ((self.tp + self.fp) / n) + (
            ((self.fp + self.tn) / n) * ((self.fn + self.tn) / n)
        )
        if p_e == 1.0:
            return 1.0
        return (p_o - p_e) / (1 - p_e)

    @property
    def agreement(self) -> float:
        n = self.tp + self.fn + self.fp + self.tn
        return (self.tp + self.tn) / n if n else 0.0

    def kappa_ci(self, iterations: int = 2000, seed: int = 7) -> tuple[float, float]:
        """Bootstrap 95% interval for kappa.

        With 24 labels the interval is wide enough that crossing a 0.60
        threshold can be sampling noise. `judge.compute_kappa` already refuses
        fewer than 20 labels and its docstring says forty or more makes the
        estimate reasonably stable; this reports the uncertainty rather than
        leaving a single decimal to carry a decision.
        """
        import random

        population = (
            [(True, True)] * self.tp
            + [(True, False)] * self.fn
            + [(False, True)] * self.fp
            + [(False, False)] * self.tn
        )
        if len(population) < 2:
            return (0.0, 0.0)
        rng = random.Random(seed)
        draws: list[float] = []
        for _ in range(iterations):
            sample = [rng.choice(population) for _ in population]
            cell = CellSummary(self.contract, self.model)
            for human, judge in sample:
                if human and judge:
                    cell.tp += 1
                elif human and not judge:
                    cell.fn += 1
                elif not human and judge:
                    cell.fp += 1
                else:
                    cell.tn += 1
            draws.append(cell.kappa)
        draws.sort()
        lo = draws[int(0.025 * len(draws))]
        hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
        return (round(lo, 4), round(hi, 4))

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
        lo, hi = self.kappa_ci()
        return {
            "contract": self.contract,
            "model": self.model,
            "kappa": round(self.kappa, 4),
            "kappa_95ci": f"[{lo:.2f}, {hi:.2f}]",
            "agreement": f"{self.tp + self.tn}/{self.tp + self.fn + self.fp + self.tn}",
            "FP": self.fp,
            "FN": self.fn,
            "good_pass": f"{self.good_passed}/{self.n_good}",
            "bad_caught": f"{self.bad_failed}/{self.n_bad}",
            "balanced_acc": round(self.balanced_accuracy, 3),
            "errors": self.errors,
            **self.per_label,
        }


@dataclass(frozen=True, slots=True)
class Cell:
    """One experimental condition. Duplicates differ only in `replicate`."""

    contract: str
    model: str
    structure: str = "current"
    replicate: int = 1


def run_grid(
    pairs: list[FrozenPair],
    contracts: tuple[str, ...] = ("v1", "v2"),
    models: tuple[str, ...] = (),
    repeats: int = 5,
    structures: tuple[str, ...] = ("current",),
    duplicate: str | None = None,
    progress=None,
) -> list[JudgeRun]:
    """Grade every frozen pair in every requested condition.

    `duplicate` names a "contract/structure" whose cell is run a second time with
    everything identical -- same pairs, contract, model, structure, repeats,
    temperature 0, no generation anywhere. Its only purpose is to measure how far
    the numbers move when nothing changed, which is the yardstick any structural
    effect has to beat. It is kept as a separate cell and never merged into the
    first; averaging the two would hide exactly what it was run to expose.
    """
    s = get_settings()
    models = models or (s.judge_model,)

    cells: list[Cell] = []
    for contract in contracts:
        if contract not in CONTRACTS:
            raise ValueError(f"unknown contract {contract!r}")
        for structure in structures:
            for model in models:
                cells.append(Cell(contract, model, structure, replicate=1))

    if duplicate:
        contract, _, structure = duplicate.partition("/")
        structure = structure or "current"
        matches = [
            c for c in cells if c.contract == contract and c.structure == structure
        ]
        if not matches:
            raise ValueError(
                f"cannot duplicate {duplicate!r}: no such cell in this grid "
                f"({sorted({f'{c.contract}/{c.structure}' for c in cells})})"
            )
        cells.extend(
            Cell(c.contract, c.model, c.structure, replicate=2) for c in matches
        )

    runs: list[JudgeRun] = []
    for cell in cells:
        for pair in pairs:
            for i in range(repeats):
                run = JudgeRun(
                    pair_id=pair.pair_id,
                    label=pair.label,
                    contract=cell.contract,
                    model=cell.model,
                    repeat=i + 1,
                    expected_verdict=pair.expected_verdict,
                    structure=cell.structure,
                    replicate=cell.replicate,
                    context_supplied=bool(pair.source_context),
                )
                try:
                    verdict = judge_narrative(
                        pair.as_case(),
                        pair.answer,
                        contract=cell.contract,
                        source_context=pair.source_context,
                        model=cell.model,
                        structure=cell.structure,
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
            if r.verdict:
                cell.tp += 1
            else:
                cell.fn += 1
        else:
            cell.n_bad += 1
            cell.bad_failed += int(not r.verdict)
            if r.verdict:
                cell.fp += 1
            else:
                cell.tn += 1
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


def majority_verdicts(runs: list[JudgeRun]) -> dict[tuple, bool]:
    """One verdict per (contract, structure, model, replicate, pair), by majority.

    Structure and replicate are in the key because they are conditions, not
    noise. Collapsing them would merge a duplicate-control cell into the cell it
    exists to be compared against.
    """
    tally: dict[tuple, list[int]] = {}
    for r in runs:
        if r.error or r.verdict is None:
            continue
        key = (r.contract, r.structure, r.model, r.replicate, r.pair_id)
        tally.setdefault(key, [0, 0])[int(bool(r.verdict))] += 1
    return {key: passes > fails for key, (fails, passes) in tally.items()}


def _resolve_cell(spec: str, runs: list[JudgeRun]) -> tuple[str, str]:
    """`v3_1` or `v3_1/reference_last` -> (contract, structure).

    A bare contract means "the only structure in this grid", which keeps the
    contract-vs-contract comparisons written before structures existed working
    unchanged.
    """
    contract, _, structure = spec.partition("/")
    if structure:
        return contract, structure
    present = {r.structure for r in runs if r.contract == contract} or {"current"}
    return contract, sorted(present)[0]


def flips(
    runs: list[JudgeRun],
    baseline: str,
    candidate: str,
    model: str | None = None,
    replicate: int = 1,
) -> list[dict[str, Any]]:
    """Which individual cases changed verdict between two conditions.

    `baseline` and `candidate` are `contract` or `contract/structure`.

    With a handful of errors out of a few dozen pairs, the aggregate cannot tell
    a real fix from a lucky one. A condition that flips the wrong cases and one
    that flips the right cases land on the same kappa; only the per-case table
    distinguishes them.
    """
    human: dict[str, bool] = {}
    for r in runs:
        human.setdefault(r.pair_id, r.expected_verdict)

    model = model or next((r.model for r in runs), "")
    base_contract, base_structure = _resolve_cell(baseline, runs)
    cand_contract, cand_structure = _resolve_cell(candidate, runs)
    decided = majority_verdicts(runs)

    rows: list[dict[str, Any]] = []
    for pair_id, expected in human.items():
        before = decided.get(
            (base_contract, base_structure, model, replicate, pair_id)
        )
        after = decided.get(
            (cand_contract, cand_structure, model, replicate, pair_id)
        )
        if before is None or after is None or before == after:
            continue
        if not expected and before and not after:
            kind = "FIXED (false positive removed)"
        elif expected and not before and after:
            kind = "FIXED (false negative removed)"
        elif expected and before and not after:
            kind = "REGRESSION (new false negative)"
        else:
            kind = "REGRESSION (new false positive)"
        rows.append(
            {
                "pair_id": pair_id[:44],
                "human": "PASS" if expected else "FAIL",
                baseline: "PASS" if before else "FAIL",
                candidate: "PASS" if after else "FAIL",
                "effect": kind,
            }
        )
    return sorted(rows, key=lambda r: ("REGRESSION" not in r["effect"], r["pair_id"]))


@dataclass(slots=True)
class PairConfusion:
    """Confusion counted once per labelled pair, not once per repeat.

    n is the number of labelled pairs, whatever the repeat count. Counting each
    repeat as its own observation multiplies n by the repeats and adds no
    information: the unit a human labelled is the answer, and repeats are the
    same answer measured again. Doing it that way narrows the confidence interval
    on nothing -- measured on the 24-pair set, v2 read kappa 0.5556 with CI
    [0.37, 0.74] at the repeat level against 0.4167 with CI [0.12, 0.74] per
    pair. The wider one is the honest one.

    Repeats still earn their cost: they establish whether the judge is stable on
    an item, reported separately as `unstable_pairs`.
    """

    contract: str
    model: str
    structure: str = "current"
    replicate: int = 1
    repeats: int = 0
    context_supplied: bool = False
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0
    unstable_pairs: list[str] = field(default_factory=list)
    verdicts: dict[str, bool] = field(default_factory=dict)

    @property
    def cell_id(self) -> str:
        """The condition, as one readable string. Duplicates differ only here."""
        return f"{self.contract}/{self.structure}/{self.model}/r{self.replicate}"

    @property
    def n(self) -> int:
        return self.tp + self.fn + self.fp + self.tn

    @property
    def kappa(self) -> float:
        cell = CellSummary(self.contract, self.model)
        cell.tp, cell.fn, cell.fp, cell.tn = self.tp, self.fn, self.fp, self.tn
        return cell.kappa

    @property
    def agreement(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    def kappa_ci(self, iterations: int = 2000) -> tuple[float, float]:
        cell = CellSummary(self.contract, self.model)
        cell.tp, cell.fn, cell.fp, cell.tn = self.tp, self.fn, self.fp, self.tn
        return cell.kappa_ci(iterations)

    def row(self) -> dict[str, Any]:
        lo, hi = self.kappa_ci()
        return {
            "cell": self.cell_id,
            "structure": self.structure,
            "rep": self.replicate,
            "repeats": self.repeats,
            "ctx": "yes" if self.context_supplied else "no",
            "n": self.n,
            "kappa": round(self.kappa, 4),
            "kappa_95ci": f"[{lo:.2f}, {hi:.2f}]",
            "agreement": f"{self.tp + self.tn}/{self.n}",
            "FP": self.fp,
            "FN": self.fn,
            "unstable": len(self.unstable_pairs),
        }


def confusion_by_pair(runs: list[JudgeRun]) -> list[PairConfusion]:
    """Per-pair majority verdicts, then the confusion against human labels.

    Keyed by the full condition including `structure` and `replicate`, so a
    duplicate-control cell stays a separate row rather than being averaged into
    the cell it exists to be compared against.
    """
    human: dict[str, bool] = {}
    for r in runs:
        human.setdefault(r.pair_id, r.expected_verdict)

    decided = majority_verdicts(runs)
    repeats_seen: dict[tuple, int] = {}
    context_seen: dict[tuple, bool] = {}
    unanimous: dict[tuple, bool] = {}
    tallies: dict[tuple, list[int]] = {}
    for r in runs:
        key = (r.contract, r.structure, r.model, r.replicate)
        repeats_seen[key] = max(repeats_seen.get(key, 0), r.repeat)
        context_seen[key] = context_seen.get(key, False) or r.context_supplied
        if r.error or r.verdict is None:
            continue
        tallies.setdefault((*key, r.pair_id), [0, 0])[int(bool(r.verdict))] += 1

    out: dict[tuple, PairConfusion] = {}
    for (contract, structure, model, replicate, pair_id), (fails, passes) in tallies.items():
        key = (contract, structure, model, replicate)
        cell = out.setdefault(
            key,
            PairConfusion(
                contract=contract,
                model=model,
                structure=structure,
                replicate=replicate,
                repeats=repeats_seen.get(key, 0),
                context_supplied=context_seen.get(key, False),
            ),
        )
        verdict_here = decided[(contract, structure, model, replicate, pair_id)]
        cell.verdicts[pair_id] = verdict_here
        if fails and passes:
            cell.unstable_pairs.append(pair_id)
        expected = human.get(pair_id, True)
        if expected and verdict_here:
            cell.tp += 1
        elif expected and not verdict_here:
            cell.fn += 1
        elif not expected and verdict_here:
            cell.fp += 1
        else:
            cell.tn += 1
    return sorted(
        out.values(), key=lambda c: (c.contract, c.structure, c.model, c.replicate)
    )


def duplicate_variance(cells: list[PairConfusion]) -> list[dict[str, Any]]:
    """How far identical conditions moved. The yardstick, not a footnote.

    Any effect smaller than this is not an effect. Reported per metric and as the
    number of pairs whose majority verdict flipped between the two executions --
    the most legible form, because those flips happened with nothing changed.
    """
    groups: dict[tuple[str, str, str], list[PairConfusion]] = {}
    for c in cells:
        groups.setdefault((c.contract, c.model, c.structure), []).append(c)

    rows: list[dict[str, Any]] = []
    for (contract, model, structure), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda c: c.replicate)
        a, b = members[0], members[1]
        shared = set(a.verdicts) & set(b.verdicts)
        flipped = sorted(p for p in shared if a.verdicts[p] != b.verdicts[p])
        rows.append(
            {
                "condition": f"{contract}/{structure}/{model}",
                "kappa_r1": round(a.kappa, 4),
                "kappa_r2": round(b.kappa, 4),
                "d_kappa": round(abs(b.kappa - a.kappa), 4),
                "FP": f"{a.fp} -> {b.fp}",
                "FN": f"{a.fn} -> {b.fn}",
                "agreement": f"{a.tp + a.tn}/{a.n} -> {b.tp + b.tn}/{b.n}",
                "verdicts_flipped": len(flipped),
                "flipped_pairs": ", ".join(flipped[:6]) or "-",
            }
        )
    return rows


def beats_noise(
    comparison_delta: float, variance_rows: list[dict[str, Any]]
) -> tuple[bool, str]:
    """Is an observed difference larger than the measured duplicate-run spread?

    The question this checkpoint exists to answer. Without a duplicate control
    there is no denominator and every delta looks like a finding.
    """
    if not variance_rows:
        return False, (
            "No duplicate-control cell was run, so there is no measured noise "
            "floor and no delta can be called an effect. Re-run with --duplicate."
        )
    noise = max(row["d_kappa"] for row in variance_rows)
    flips_seen = max(row["verdicts_flipped"] for row in variance_rows)
    if abs(comparison_delta) > noise:
        return True, (
            f"delta kappa {comparison_delta:+.4f} exceeds the duplicate-run spread "
            f"of {noise:.4f} ({flips_seen} pair verdict(s) moved with nothing changed)."
        )
    return False, (
        f"delta kappa {comparison_delta:+.4f} is within the duplicate-run spread of "
        f"{noise:.4f} ({flips_seen} pair verdict(s) moved with nothing changed). "
        "UNRESOLVED -- this is not evidence of an effect."
    )


# --------------------------------------------------------- adoption criteria
DEFAULT_CRITERIA: dict[str, Any] = {
    "max_new_false_negatives": 0,
    "max_false_positives": 4,
    "min_kappa": 0.60,
    "must_pass": ["nar-PG-fx-exposure", "nar-PG-legal"],
    "must_fail": [
        "nar-CAT-fx-exposure",
        "nar-AAPL-supply-concentration",
        "nar-COST-supply-concentration",
    ],
}


def check_adoption(
    runs: list[JudgeRun],
    candidate: str,
    baseline: str,
    model: str | None = None,
    criteria: dict[str, Any] | None = None,
    variance_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score a candidate condition against a declared, named acceptance gate.

    Declared before the run and checked mechanically, because an aggregate can be
    satisfied by the wrong cases. v3 reached kappa 0.6667 -- above the threshold --
    by fixing three false positives and breaking two correct answers. A
    kappa-only rule would have adopted it.

    When `variance_rows` are supplied, beating the measured duplicate-run spread
    becomes a clause of its own. A gain inside the noise floor is not a gain, and
    that judgement belongs in the gate rather than in the reader's head.
    """
    spec = {**DEFAULT_CRITERIA, **(criteria or {})}
    model = model or next((r.model for r in runs), "")
    decided = majority_verdicts(runs)
    cand_contract, cand_structure = _resolve_cell(candidate, runs)
    by_cell = {
        (c.contract, c.structure): c
        for c in confusion_by_pair(runs)
        if c.replicate == 1
    }
    cell = by_cell.get((cand_contract, cand_structure))
    base_contract, base_structure = _resolve_cell(baseline, runs)
    base_cell = by_cell.get((base_contract, base_structure))
    flip_rows = flips(runs, baseline, candidate, model)
    new_fn = [r for r in flip_rows if "new false negative" in r["effect"]]

    clauses: list[dict[str, Any]] = [
        {
            "clause": "new false negatives = 0",
            "observed": len(new_fn),
            "ok": len(new_fn) <= spec["max_new_false_negatives"],
            "detail": ", ".join(r["pair_id"] for r in new_fn) or "-",
        },
        {
            "clause": f"FP <= {spec['max_false_positives']}",
            "observed": cell.fp if cell else "-",
            "ok": bool(cell) and cell.fp <= spec["max_false_positives"],
            "detail": "-",
        },
    ]
    for pair_id in spec["must_pass"]:
        got = decided.get((cand_contract, cand_structure, model, 1, pair_id))
        clauses.append(
            {
                "clause": f"{pair_id} = PASS",
                "observed": "-" if got is None else ("PASS" if got else "FAIL"),
                "ok": got is True,
                "detail": "not graded" if got is None else "-",
            }
        )
    for pair_id in spec["must_fail"]:
        got = decided.get((cand_contract, cand_structure, model, 1, pair_id))
        clauses.append(
            {
                "clause": f"{pair_id} = FAIL",
                "observed": "-" if got is None else ("PASS" if got else "FAIL"),
                "ok": got is False,
                "detail": "not graded" if got is None else "-",
            }
        )
    kappa = cell.kappa if cell else 0.0
    lo, hi = cell.kappa_ci() if cell else (0.0, 0.0)
    clauses.append(
        {
            "clause": f"kappa >= {spec['min_kappa']}",
            "observed": round(kappa, 4),
            "ok": kappa >= spec["min_kappa"],
            "detail": f"95% CI [{lo:.2f}, {hi:.2f}]",
        }
    )

    if variance_rows is not None:
        delta = kappa - (base_cell.kappa if base_cell else 0.0)
        ok, message = beats_noise(delta, variance_rows)
        clauses.append(
            {
                "clause": "effect exceeds duplicate-run noise",
                "observed": round(delta, 4),
                "ok": ok,
                "detail": message,
            }
        )

    adopted = all(c["ok"] for c in clauses)
    failed = [c["clause"] for c in clauses if not c["ok"]]
    return {
        "candidate": candidate,
        "baseline": baseline,
        "adopted": adopted,
        "clauses": clauses,
        "failed_clauses": failed,
        "summary": (
            f"{candidate} MEETS every adoption clause."
            if adopted
            else f"{candidate} does NOT meet: {'; '.join(failed)}."
        ),
    }


def regression_guard(flip_rows: list[dict[str, Any]]) -> tuple[bool, str]:
    """A new false negative is a stop, not a line item.

    The whole reason v1 was replaced is that it failed correct answers. A v3
    that raises kappa by starting to do the same thing has not fixed the judge,
    it has moved the error to the side that is harder to notice -- a wrongly
    failed answer looks like a system bug and sends the next week into
    retrieval.
    """
    new_fn = [r for r in flip_rows if "new false negative" in r["effect"]]
    if new_fn:
        names = ", ".join(r["pair_id"] for r in new_fn)
        return False, (
            f"BLOCKED: {len(new_fn)} new false negative(s) -- {names}. A stricter "
            "rubric that starts failing correct answers is the v1 defect "
            "returning, whatever it did to kappa."
        )
    return True, "No new false negatives."


def verdict(cells: list[CellSummary], comparisons: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    if not cells:
        return "no cells run"

    labelled = [c for c in cells if (c.tp + c.fn + c.fp + c.tn) >= 20]
    if labelled:
        best = max(labelled, key=lambda c: c.kappa)
        lo, hi = best.kappa_ci()
        n = best.tp + best.fn + best.fp + best.tn
        lines.append(
            f"Best cell: contract={best.contract} model={best.model}, "
            f"kappa {best.kappa:.4f} (95% CI [{lo:.2f}, {hi:.2f}], n={n}), "
            f"agreement {best.tp + best.tn}/{n}, FP={best.fp}, FN={best.fn}."
        )
        if best.kappa >= 0.60 and lo < 0.60:
            lines.append(
                f"kappa clears 0.60 but its interval reaches {lo:.2f}. On {n} labels "
                "that threshold crossing is not by itself decisive -- read the "
                "false-positive count and the per-case flips, which do not depend "
                "on a point estimate."
            )
    else:
        best = max(cells, key=lambda c: c.balanced_accuracy)
        lines.append(
            f"Best cell: contract={best.contract} model={best.model}, "
            f"balanced accuracy {best.balanced_accuracy:.3f} "
            f"(good {best.good_passed}/{best.n_good}, bad caught {best.bad_failed}/{best.n_bad})."
        )

    one_sided = [c for c in cells if c.fp >= 3 and c.fn == 0]
    if one_sided:
        names = ", ".join(f"{c.contract}/{c.model}" for c in one_sided)
        lines.append(
            f"ONE-SIDED: {names} makes only false-positive errors. That is a "
            "permissive judge rather than a noisy one, so the fix is a rule the "
            "rubric is missing, not a stronger model."
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


def pairs_from_labels(run_key: str) -> list[FrozenPair]:
    """Frozen pairs whose expected verdict is the HUMAN label, not a mutation.

    These are better controls than anything `build_controls` produces, and the
    24 labels proved it. The synthetic omission control -- truncate the answer to
    its first two sentences -- scored 8/8 in
    `reports/judge_v2_with_omission.json`, while real omissions slipped past the
    same judge seven times in the same period. Truncation makes an obviously
    stunted answer; a real omission is fluent, confident, and covers one part of
    a multi-part disclosure.

    So the mutations test whether the judge catches *detectable* breakage, and
    human labels test whether it catches *plausible* breakage. Only the second
    predicts behaviour in a run.

    New eval runs persist both the original question and the exact source
    context shown to the production judge. That makes human-labelled answers
    replayable under later judge contracts without changing the evidence the
    judge sees.
    """
    from .. import db

    rows = _labelled_rows(run_key)

    return [
        FrozenPair(
            pair_id=row["case_id"],
            question=row.get("question") or row["case_id"],
            reference=row["expected"] or "",
            answer=row["answer"] or "",
            source_context=row.get("context_text") or "",
            expected_verdict=bool(row["human_label"]),
            label="human-pass" if row["human_label"] else "human-fail",
            provenance=f"human label on {run_key}",
        )
        for row in rows
    ]


def _case_kind(pair_id: str) -> str:
    """The narrative topic, e.g. `nar-AAPL-fx-exposure` -> `fx-exposure`."""
    parts = pair_id.split("-", 2)
    return parts[2] if len(parts) > 2 else "unknown"


def _case_ticker(pair_id: str) -> str:
    parts = pair_id.split("-")
    return parts[1] if len(parts) > 1 else "unknown"


def composition(pairs: list[FrozenPair]) -> dict[str, Any]:
    """What the calibration set is actually made of.

    A kappa on 40 pairs that are 30 Apple legal questions is a kappa on one
    question type. Reported so a skewed set is visible before it is used as a
    yardstick, rather than inferred afterwards from a surprising result.
    """
    by_kind: dict[str, int] = {}
    by_ticker: dict[str, int] = {}
    by_label: dict[str, int] = {}
    n_pass = 0
    for pair in pairs:
        by_kind[_case_kind(pair.pair_id)] = by_kind.get(_case_kind(pair.pair_id), 0) + 1
        by_ticker[_case_ticker(pair.pair_id)] = (
            by_ticker.get(_case_ticker(pair.pair_id), 0) + 1
        )
        by_label[pair.label] = by_label.get(pair.label, 0) + 1
        n_pass += int(pair.expected_verdict)

    total = len(pairs)
    largest_kind = max(by_kind.values()) if by_kind else 0
    largest_ticker = max(by_ticker.values()) if by_ticker else 0
    warnings: list[str] = []
    if total < 40:
        warnings.append(f"only {total} labelled pairs; 40 is the floor for a stable kappa")
    if total and largest_kind / total > 0.4:
        dominant = max(by_kind, key=lambda k: by_kind[k])
        warnings.append(
            f"{dominant} is {largest_kind}/{total} of the set -- kappa will mostly "
            "describe that question type"
        )
    if total and largest_ticker / total > 0.3:
        dominant = max(by_ticker, key=lambda k: by_ticker[k])
        warnings.append(f"{dominant} is {largest_ticker}/{total} of the set")
    if total and not 0.25 <= n_pass / total <= 0.75:
        warnings.append(
            f"labels are {n_pass} PASS / {total - n_pass} FAIL -- a lopsided set "
            "raises chance agreement and shrinks the kappa a real judge can earn"
        )

    return {
        "total": total,
        "human_pass": n_pass,
        "human_fail": total - n_pass,
        "by_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
        "by_ticker": dict(sorted(by_ticker.items(), key=lambda kv: -kv[1])),
        "by_label": dict(sorted(by_label.items(), key=lambda kv: -kv[1])),
        "warnings": warnings,
    }


def label_targets(run_key: str, want: int = 40) -> list[dict[str, Any]]:
    """Which unlabelled narrative cases to label next, ordered for diversity.

    Labelling is the expensive step and it is human, so the order matters: this
    puts the case whose (kind, ticker) is least represented in the existing
    labels first. It selects *what to look at* and never what the verdict is --
    the judge's own opinion is deliberately not a tiebreak, because a set chosen
    to agree with the judge cannot measure the judge.
    """
    from .. import db
    from .judge import _hash_answer

    labelled = {
        (r["case_id"], r["answer_hash"])
        for r in db.query("SELECT case_id, answer_hash FROM judge_calibration")
    }
    rows = db.query(
        """
        SELECT r.case_id, r.answer, r.expected, r.passed
          FROM eval_results r
          JOIN eval_runs u ON u.id = r.run_id
         WHERE u.run_key = %s AND r.kind = 'narrative'
         ORDER BY r.case_id
        """,
        (run_key,),
    )

    have_kind: dict[str, int] = {}
    have_ticker: dict[str, int] = {}
    unlabelled: list[dict[str, Any]] = []
    for row in rows:
        kind, ticker = _case_kind(row["case_id"]), _case_ticker(row["case_id"])
        if (row["case_id"], _hash_answer(row["answer"] or "")) in labelled:
            have_kind[kind] = have_kind.get(kind, 0) + 1
            have_ticker[ticker] = have_ticker.get(ticker, 0) + 1
        else:
            unlabelled.append(
                {
                    "case_id": row["case_id"],
                    "kind": kind,
                    "ticker": ticker,
                    "judge_said": "PASS" if row["passed"] else "FAIL",
                }
            )

    ordered: list[dict[str, Any]] = []
    while unlabelled and len(ordered) < want:
        nxt = min(
            unlabelled,
            key=lambda c: (
                have_kind.get(c["kind"], 0),
                have_ticker.get(c["ticker"], 0),
                c["case_id"],
            ),
        )
        unlabelled.remove(nxt)
        have_kind[nxt["kind"]] = have_kind.get(nxt["kind"], 0) + 1
        have_ticker[nxt["ticker"]] = have_ticker.get(nxt["ticker"], 0) + 1
        ordered.append(nxt)
    return ordered


def _labelled_rows(run_key: str) -> list[dict[str, Any]]:
    """Join eval_results to human labels on (case_id, answer_hash).

    Matched on the answer hash rather than case_id alone, for the reason
    `judge.calibration_pairs` already gives: a human label applies to the
    specific answer that was labelled, and reusing it for a changed answer
    inflates agreement.
    """
    from .. import db
    from .judge import _hash_answer

    labels = {
        (r["case_id"], r["answer_hash"]): bool(r["human_label"])
        for r in db.query("SELECT case_id, answer_hash, human_label FROM judge_calibration")
    }
    rows = db.query(
        """
        SELECT r.case_id, r.question, r.answer, r.expected, r.context_text
          FROM eval_results r
          JOIN eval_runs u ON u.id = r.run_id
         WHERE u.run_key = %s AND r.kind = 'narrative'
         ORDER BY r.case_id
        """,
        (run_key,),
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (row["case_id"], _hash_answer(row["answer"] or ""))
        if key in labels:
            out.append({**dict(row), "human_label": labels[key]})
    return out


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
