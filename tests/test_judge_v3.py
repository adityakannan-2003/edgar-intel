"""v3, and measuring a judge against human labels rather than mutations.

24 human labels on baseline-v5 scored v2 at kappa 0.4167: agreement 17/24 with
**7 false positives and 0 false negatives**. Perfectly one-sided. And supplying
the source context had *raised* v2's pass rate from 50% to 79% -- every one of
those extra passes was wrong, because with passages to check against v2 stopped
failing extra detail while still never checking completeness or scope.

The mutation controls had said everything was fine: omission 8/8 in
`reports/judge_v2_with_omission.json`, on the same judge, in the same period
that real omissions slipped past seven times. Truncating an answer produces an
obviously stunted one; a real omission is fluent and covers one part of a
multi-part disclosure. That gap is the reason these tests exist.
"""

from __future__ import annotations

import pytest

from edgar_intel.evals.judge_contracts import CONTRACTS, get_contract
from edgar_intel.evals.judge_lab import (
    CellSummary,
    JudgeRun,
    flips,
    majority_verdicts,
    regression_guard,
    summarise,
    verdict,
)


class TestV2IsFrozen:
    """v2 becomes the control arm in turn. It must not drift."""

    def test_v2_system_is_unchanged(self):
        v2 = CONTRACTS["v2"].system
        assert "1. Contradiction." in v2
        assert "2. Required coverage." in v2
        assert "3. Unsupported material claims." in v2
        assert "MATERIAL COMPLETENESS" not in v2
        assert "SCOPE FIDELITY" not in v2

    def test_v1_is_still_untouched(self):
        assert "Extra correct detail is fine" in CONTRACTS["v1"].system
        assert not CONTRACTS["v1"].wants_source_context

    def test_every_prior_contract_survives_a_new_one(self):
        """Old contracts are provenance. A run recorded under v2 stays explicable."""
        assert {"v1", "v2", "v3"} <= set(CONTRACTS)


class TestV3AddsTheTwoMissingRules:
    def test_material_completeness_is_present(self):
        system = CONTRACTS["v3"].system
        assert "MATERIAL COMPLETENESS" in system
        assert "covers only one part of a multi-part disclosure" in system

    def test_completeness_does_not_demand_every_detail(self):
        """Otherwise v3 recreates v1: failing answers for missing minor detail."""
        system = CONTRACTS["v3"].system
        assert "Do not treat every reference detail as mandatory" in system
        assert "Distinguish supporting detail from central answer content" in system

    def test_scope_fidelity_is_present(self):
        system = CONTRACTS["v3"].system
        assert "SCOPE FIDELITY" in system
        assert "Company-wide results must not be replaced by segment-level results" in system
        assert "Adjacent risks are not substitutes" in system

    def test_the_verdict_instruction_is_present(self):
        """The sentence aimed squarely at the 7 false positives."""
        system = CONTRACTS["v3"].system
        assert "Do not PASS merely because every sentence in the candidate is true." in system
        assert "materially and at the correct scope" in system

    def test_the_four_step_procedure_is_ordered(self):
        system = CONTRACTS["v3"].system
        central = system.index("Identify the central claims")
        covers = system.index("Check whether the candidate covers")
        scope = system.index("Check whether company/segment")
        unsupported = system.index("Only then evaluate unsupported")
        assert central < covers < scope < unsupported

    def test_v3_keeps_what_v2_fixed(self):
        """A stricter rubric must not bring back v1's reasoning."""
        system = CONTRACTS["v3"].system
        assert '"Not mentioned in the reference" does not mean "incorrect."' in system
        assert "Paraphrases and equivalent terminology are acceptable" in system
        assert "Extra detail is allowed" in system

    def test_v3_still_wants_source_context(self):
        assert CONTRACTS["v3"].wants_source_context
        assert "SOURCE CONTEXT" in CONTRACTS["v3"].template

    def test_v3_shares_v2s_template(self):
        """One variable per comparison: v2 -> v3 changes the rubric, not the layout."""
        assert CONTRACTS["v3"].template == CONTRACTS["v2"].template


class TestKappaAgainstHumanLabels:
    def _cell(self, tp: int, fn: int, fp: int, tn: int) -> CellSummary:
        cell = CellSummary("v2", "gpt-4o-mini")
        cell.tp, cell.fn, cell.fp, cell.tn = tp, fn, fp, tn
        return cell

    def test_reproduces_the_measured_v2_kappa(self):
        """The real numbers: 12 / 0 / 7 / 5 on baseline-v5."""
        cell = self._cell(12, 0, 7, 5)
        assert cell.kappa == pytest.approx(0.4167, abs=1e-4)
        assert cell.agreement == pytest.approx(17 / 24, abs=1e-6)

    def test_kappa_punishes_a_skewed_judge_that_agreement_flatters(self):
        """70.8% agreement reads fine; kappa says half of it was chance."""
        cell = self._cell(12, 0, 7, 5)
        assert cell.agreement > 0.70
        assert cell.kappa < 0.45

    def test_perfect_agreement(self):
        assert self._cell(12, 0, 0, 12).kappa == pytest.approx(1.0)

    def test_a_judge_that_passes_everything_scores_zero(self):
        assert self._cell(12, 0, 12, 0).kappa == pytest.approx(0.0)

    def test_unanimous_identical_raters_return_one_not_nan(self):
        assert self._cell(24, 0, 0, 0).kappa == 1.0

    def test_empty_cell(self):
        assert self._cell(0, 0, 0, 0).kappa == 0.0

    def test_bootstrap_interval_brackets_the_estimate(self):
        cell = self._cell(12, 0, 7, 5)
        lo, hi = cell.kappa_ci(iterations=400)
        assert lo <= cell.kappa <= hi

    def test_the_interval_on_24_labels_is_wide(self):
        """Why a 0.60 threshold crossing is not decisive on this sample."""
        lo, hi = self._cell(12, 0, 7, 5).kappa_ci(iterations=400)
        assert hi - lo > 0.25

    def test_row_reports_kappa_and_the_confusion_counts(self):
        row = self._cell(12, 0, 7, 5).row()
        assert row["kappa"] == pytest.approx(0.4167, abs=1e-4)
        assert row["FP"] == 7
        assert row["FN"] == 0
        assert row["agreement"] == "17/24"


class TestOneSidedErrorIsNamed:
    def _runs(self, contract: str, fp: int, fn: int, tp: int, tn: int) -> list[JudgeRun]:
        runs = []
        for i in range(tp):
            runs.append(JudgeRun(f"tp{i}", "human-pass", contract, "m", 1, True, verdict=True))
        for i in range(fn):
            runs.append(JudgeRun(f"fn{i}", "human-pass", contract, "m", 1, True, verdict=False))
        for i in range(fp):
            runs.append(JudgeRun(f"fp{i}", "human-fail", contract, "m", 1, False, verdict=True))
        for i in range(tn):
            runs.append(JudgeRun(f"tn{i}", "human-fail", contract, "m", 1, False, verdict=False))
        return runs

    def test_permissive_judge_is_called_one_sided(self):
        cells = summarise(self._runs("v2", fp=7, fn=0, tp=12, tn=5))
        text = verdict(cells, [])
        assert "ONE-SIDED" in text
        assert "permissive judge rather than a noisy one" in text

    def test_a_balanced_error_profile_is_not_flagged(self):
        cells = summarise(self._runs("v3", fp=2, fn=2, tp=10, tn=10))
        assert "ONE-SIDED" not in verdict(cells, [])

    def test_verdict_reports_kappa_when_labels_are_present(self):
        cells = summarise(self._runs("v2", fp=7, fn=0, tp=12, tn=5))
        assert "kappa" in verdict(cells, [])

    def test_a_marginal_threshold_crossing_is_qualified(self):
        cells = summarise(self._runs("v3", fp=2, fn=1, tp=11, tn=10))
        text = verdict(cells, [])
        if cells[0].kappa >= 0.60:
            assert "not by itself decisive" in text or "interval reaches" in text


class TestPerCaseFlips:
    """With 7 errors out of 24, which cases moved matters more than the total."""

    def _pair_runs(self, pair_id: str, human: bool, v2: bool, v3: bool) -> list[JudgeRun]:
        return [
            JudgeRun(pair_id, "x", "v2", "m", 1, human, verdict=v2),
            JudgeRun(pair_id, "x", "v3", "m", 1, human, verdict=v3),
        ]

    def test_a_removed_false_positive_is_named_as_fixed(self):
        runs = self._pair_runs("nar-PG-revenue-drivers", human=False, v2=True, v3=False)
        rows = flips(runs, "v2", "v3", "m")
        assert len(rows) == 1
        assert "FIXED" in rows[0]["effect"]
        assert "false positive removed" in rows[0]["effect"]

    def test_a_new_false_negative_is_named_as_a_regression(self):
        runs = self._pair_runs("nar-MSFT-legal", human=True, v2=True, v3=False)
        rows = flips(runs, "v2", "v3", "m")
        assert "REGRESSION" in rows[0]["effect"]
        assert "new false negative" in rows[0]["effect"]

    def test_unchanged_cases_are_not_listed(self):
        runs = self._pair_runs("stable", human=True, v2=True, v3=True)
        assert flips(runs, "v2", "v3", "m") == []

    def test_regressions_sort_first(self):
        runs = (
            self._pair_runs("a-fixed", human=False, v2=True, v3=False)
            + self._pair_runs("z-broken", human=True, v2=True, v3=False)
        )
        rows = flips(runs, "v2", "v3", "m")
        assert "REGRESSION" in rows[0]["effect"]

    def test_majority_decides_across_repeats(self):
        runs = [
            JudgeRun("p", "x", "v3", "m", 1, False, verdict=False),
            JudgeRun("p", "x", "v3", "m", 2, False, verdict=False),
            JudgeRun("p", "x", "v3", "m", 3, False, verdict=True),
        ]
        assert majority_verdicts(runs)[("v3", "m", "p")] is False

    def test_errored_repeats_are_ignored_not_counted_as_fail(self):
        runs = [
            JudgeRun("p", "x", "v3", "m", 1, True, verdict=True),
            JudgeRun("p", "x", "v3", "m", 2, True, error="400"),
        ]
        assert majority_verdicts(runs)[("v3", "m", "p")] is True


class TestRegressionGuard:
    """A stricter rubric that starts failing correct answers is v1 returning."""

    def test_a_new_false_negative_blocks(self):
        rows = [{"pair_id": "nar-MSFT-legal", "effect": "REGRESSION (new false negative)"}]
        ok, message = regression_guard(rows)
        assert not ok
        assert "BLOCKED" in message
        assert "nar-MSFT-legal" in message

    def test_removing_false_positives_passes(self):
        rows = [
            {"pair_id": "nar-PG-revenue-drivers", "effect": "FIXED (false positive removed)"},
            {"pair_id": "nar-COST-supply", "effect": "FIXED (false positive removed)"},
        ]
        ok, message = regression_guard(rows)
        assert ok
        assert "No new false negatives" in message

    def test_no_flips_at_all_passes(self):
        ok, _ = regression_guard([])
        assert ok

    def test_the_guard_does_not_care_about_kappa(self):
        """It is a veto, not a score. A kappa gain cannot buy a false negative."""
        rows = [
            {"pair_id": f"fixed{i}", "effect": "FIXED (false positive removed)"}
            for i in range(7)
        ] + [{"pair_id": "broken", "effect": "REGRESSION (new false negative)"}]
        ok, _ = regression_guard(rows)
        assert not ok


class TestHumanLabelsOutrankMutations:
    def test_labelled_pairs_carry_the_human_verdict(self):
        from edgar_intel.evals.judge_lab import FrozenPair

        pair = FrozenPair(
            pair_id="nar-PG-revenue-drivers",
            question="q",
            reference="overall unit volume unchanged, organic growth 1%",
            answer="Beauty segment net sales rose 2%...",
            expected_verdict=False,
            label="human-fail",
            provenance="human label on baseline-v5",
        )
        assert pair.expected_verdict is False
        assert "human" in pair.label

    def test_the_default_contract_is_not_silently_changed(self):
        """v3 must be measured before it ships, like v2 was supposed to be."""
        from edgar_intel.evals.judge_contracts import DEFAULT_CONTRACT

        assert DEFAULT_CONTRACT in {"v2", "v3"}
        assert get_contract().name == DEFAULT_CONTRACT
