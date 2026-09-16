"""Judge calibration: contracts as variables, and negative controls.

The temptation after finding that v1 fails a good answer is to write a gentler
rubric, watch the good answer pass, and declare the judge fixed. That measures
permissiveness, not correctness -- a judge that passes everything scores 100% on
that test and is worth nothing. Every test here exists to keep both halves of
the measurement attached to each other.
"""

from __future__ import annotations

import pytest

from edgar_intel.evals.judge_contracts import CONTRACTS, DEFAULT_CONTRACT, get_contract
from edgar_intel.evals.judge_lab import (
    CellSummary,
    FrozenPair,
    JudgeRun,
    build_controls,
    mutate_contradiction,
    mutate_fabrication,
    mutate_hedge,
    mutate_omission,
    single_variable_comparisons,
    summarise,
    verdict,
)

GOOD_ANSWER = (
    "Apple Inc. discloses several material legal proceedings, including: 1) A "
    "violation of a 2021 Injunction by the California District Court. 2) The "
    "Company is subject to investigations under the Digital Markets Act (DMA) by "
    "the European Commission, which has fined the Company €500 million. 3) A civil "
    "antitrust lawsuit filed by the Department of Justice and various state "
    "attorneys general. 4) A lawsuit filed by Epic Games related to the App Store."
)

PAIR = FrozenPair(
    pair_id="nar-AAPL-legal",
    question="What material legal proceedings does Apple Inc. disclose?",
    reference="Apple discloses EU Digital Markets Act proceedings, including a €500 "
    "million fine ... and the 2021 injunction.",
    answer=GOOD_ANSWER,
    source_context="[1] AAPL FY2025 Item 3 ... Digital Markets Act ... €500 million ...",
)


class TestContractsAreVersioned:
    def test_v1_is_still_the_shipped_default(self):
        """A rubric change before it is measured makes every earlier run incomparable."""
        assert DEFAULT_CONTRACT == "v1"
        assert get_contract().name == "v1"

    def test_v1_is_preserved_as_the_control_arm(self):
        v1 = CONTRACTS["v1"]
        assert "Extra correct detail is fine" in v1.system
        assert not v1.wants_source_context

    def test_v2_asks_for_source_context(self):
        """The clause v1 could not check becomes checkable only with the passages."""
        v2 = CONTRACTS["v2"]
        assert v2.wants_source_context
        assert "SOURCE CONTEXT" in v2.template

    def test_v2_names_the_three_failure_modes(self):
        system = CONTRACTS["v2"].system
        assert "Contradiction" in system
        assert "Required coverage" in system
        assert "Unsupported material claims" in system

    def test_v2_rejects_the_reasoning_that_failed_the_apple_case(self):
        assert '"Not mentioned in the reference" does not mean "incorrect."' in CONTRACTS["v2"].system

    def test_v2_still_rejects_hedges(self):
        """Loosening the rubric must not license a generic non-answer."""
        assert "hedge" in CONTRACTS["v2"].system.lower()

    def test_rendering_fills_every_slot(self):
        rendered = CONTRACTS["v2"].render("Q?", "REF", "CAND", "CTX")
        for token in ("Q?", "REF", "CAND", "CTX"):
            assert token in rendered

    def test_missing_context_is_marked_not_silently_blank(self):
        assert "(not supplied)" in CONTRACTS["v2"].render("Q", "R", "C", "")

    def test_unknown_contract_raises(self):
        with pytest.raises(ValueError):
            get_contract("v99")


class TestNegativeControls:
    """Each mutation targets exactly one rubric dimension."""

    def test_contradiction_changes_a_figure(self):
        bad = mutate_contradiction(PAIR)
        assert bad is not None
        assert bad.expected_verdict is False
        assert "€500" not in bad.answer
        assert bad.label == "contradiction"

    def test_contradiction_returns_none_without_a_figure(self):
        plain = FrozenPair("x", "q", "r", "no figures here at all")
        assert mutate_contradiction(plain) is None

    def test_omission_drops_material_coverage(self):
        bad = mutate_omission(PAIR)
        assert bad is not None
        assert "Epic Games" not in bad.answer
        assert len(bad.answer) < len(PAIR.answer)
        assert bad.expected_verdict is False

    def test_omission_returns_none_when_too_short_to_cut(self):
        short = FrozenPair("x", "q", "r", "One sentence.")
        assert mutate_omission(short) is None

    def test_hedge_is_a_generic_non_answer(self):
        bad = mutate_hedge(PAIR)
        assert "Epic Games" not in bad.answer
        assert "Digital Markets Act" not in bad.answer
        assert bad.expected_verdict is False

    def test_fabrication_adds_something_in_neither_reference_nor_context(self):
        bad = mutate_fabrication(PAIR)
        assert "securities class action" in bad.answer
        assert "securities class action" not in PAIR.reference
        assert "securities class action" not in PAIR.source_context
        assert bad.expected_verdict is False

    def test_fabrication_keeps_the_correct_content(self):
        """It must fail for the fabrication, not for losing the right answer."""
        bad = mutate_fabrication(PAIR)
        assert "Epic Games" in bad.answer
        assert "€500 million" in bad.answer

    def test_build_controls_covers_every_dimension(self):
        labels = {p.label for p in build_controls(PAIR)}
        assert labels == {"good", "contradiction", "omission", "hedge", "fabrication"}

    def test_controls_carry_provenance(self):
        for p in build_controls(PAIR):
            if p.label != "good":
                assert PAIR.pair_id in p.provenance


class TestScoringNeedsBothHalves:
    def _runs(self, contract: str, model: str, good_pass: int, bad_fail: int,
              n_good: int = 5, n_bad: int = 20) -> list[JudgeRun]:
        runs = []
        for i in range(n_good):
            runs.append(JudgeRun("p", "good", contract, model, i, True,
                                 verdict=i < good_pass, correct=i < good_pass))
        for i in range(n_bad):
            failed = i < bad_fail
            runs.append(JudgeRun("p::bad", "contradiction", contract, model, i, False,
                                 verdict=not failed, correct=failed))
        return runs

    def test_a_permissive_judge_does_not_score_well(self):
        """Passes every good answer, catches nothing. The failure mode to avoid."""
        cell = summarise(self._runs("v2", "mini", good_pass=5, bad_fail=0))[0]
        assert cell.good_pass_rate == 1.0
        assert cell.bad_catch_rate == 0.0
        assert cell.balanced_accuracy == 0.5

    def test_a_strict_judge_does_not_score_well(self):
        """v1's shape: catches everything bad, fails the good answer too."""
        cell = summarise(self._runs("v1", "mini", good_pass=0, bad_fail=20))[0]
        assert cell.balanced_accuracy == 0.5

    def test_only_doing_both_scores_well(self):
        cell = summarise(self._runs("v2", "mini", good_pass=5, bad_fail=19))[0]
        assert cell.balanced_accuracy > 0.9

    def test_permissive_cells_are_called_out_by_name(self):
        cells = summarise(self._runs("v2", "mini", good_pass=5, bad_fail=2))
        text = verdict(cells, [])
        assert "PERMISSIVE" in text
        assert "worse instrument" in text

    def test_strict_cells_are_called_out(self):
        cells = summarise(self._runs("v1", "mini", good_pass=1, bad_fail=20))
        assert "STRICT" in verdict(cells, [])

    def test_errors_do_not_count_as_verdicts(self):
        runs = [JudgeRun("p", "good", "v2", "mini", 1, True, error="400")]
        cell = summarise(runs)[0]
        assert cell.errors == 1
        assert cell.n_good == 0


class TestSingleVariableComparisons:
    """Changing rubric and model together proves nothing."""

    def _cell(self, contract: str, model: str, acc: float) -> CellSummary:
        cell = CellSummary(contract, model)
        cell.n_good, cell.good_passed = 10, int(acc * 10)
        cell.n_bad, cell.bad_failed = 10, int(acc * 10)
        return cell

    def test_same_model_different_contract_is_a_prompt_effect(self):
        cells = [self._cell("v1", "mini", 0.4), self._cell("v2", "mini", 0.9)]
        comps = single_variable_comparisons(cells)
        assert len(comps) == 1
        assert comps[0]["varies"] == "contract"

    def test_same_contract_different_model_is_a_model_effect(self):
        cells = [self._cell("v1", "mini", 0.4), self._cell("v1", "gpt-4o", 0.8)]
        assert single_variable_comparisons(cells)[0]["varies"] == "model"

    def test_both_varying_is_excluded(self):
        """The confound this module exists to prevent."""
        cells = [self._cell("v1", "mini", 0.4), self._cell("v2", "gpt-4o", 0.95)]
        assert single_variable_comparisons(cells) == []

    def test_a_full_grid_yields_only_the_clean_pairs(self):
        cells = [
            self._cell("v1", "mini", 0.4),
            self._cell("v2", "mini", 0.9),
            self._cell("v1", "gpt-4o", 0.6),
            self._cell("v2", "gpt-4o", 0.95),
        ]
        comps = single_variable_comparisons(cells)
        assert len(comps) == 4
        assert {c["varies"] for c in comps} == {"contract", "model"}

    def test_verdict_names_the_dominant_factor(self):
        cells = [
            self._cell("v1", "mini", 0.4),
            self._cell("v2", "mini", 0.9),
            self._cell("v1", "gpt-4o", 0.5),
            self._cell("v2", "gpt-4o", 0.95),
        ]
        text = verdict(cells, single_variable_comparisons(cells))
        assert "prompt is the dominant factor" in text.lower()

    def test_verdict_says_so_when_the_model_dominates(self):
        cells = [
            self._cell("v1", "mini", 0.4),
            self._cell("v2", "mini", 0.45),
            self._cell("v1", "gpt-4o", 0.9),
            self._cell("v2", "gpt-4o", 0.92),
        ]
        text = verdict(cells, single_variable_comparisons(cells))
        assert "rubric rewrite will not be enough" in text


class TestJudgeSignature:
    def test_contract_and_model_are_separate_parameters(self):
        """One knob per comparison is only possible if they are separate knobs."""
        import inspect

        from edgar_intel.evals.judge import judge_narrative

        params = inspect.signature(judge_narrative).parameters
        assert "contract" in params
        assert "model" in params
        assert "source_context" in params

    def test_default_contract_is_unchanged(self):
        import inspect

        from edgar_intel.evals.judge import judge_narrative

        assert inspect.signature(judge_narrative).parameters["contract"].default is None
