"""v3_1, and an adoption gate that a good kappa cannot buy its way past.

v3 scored kappa 0.6667 on the 24 human labels -- above the 0.60 target -- and
was blocked. It fixed three false positives (Apple supply, CAT FX, Costco
supply) and created two false negatives (P&G FX, P&G legal): correct answers
whose references simply carried more supporting detail. A kappa-only rule would
have adopted it.

MATERIAL COMPLETENESS told the judge an incomplete answer fails and never said
where completeness stops. v3_1 adds that boundary, and the tests below hold the
delta to exactly two insertions so v3 stays a usable control.
"""

from __future__ import annotations

import pytest

from edgar_intel.evals.judge_contracts import (
    CONTRACTS,
    _V3_1_DO_NOT_FAIL,
    _V3_1_MATERIALITY_BOUNDARY,
    get_contract,
)
from edgar_intel.evals.judge_lab import (
    JudgeRun,
    check_adoption,
    confusion_by_pair,
)


class TestV3IsFrozen:
    def test_v3_has_no_materiality_boundary(self):
        assert "MATERIALITY BOUNDARY" not in CONTRACTS["v3"].system
        assert "Do not FAIL a substantively correct answer" not in CONTRACTS["v3"].system

    def test_v3_keeps_its_own_rules(self):
        system = CONTRACTS["v3"].system
        assert "MATERIAL COMPLETENESS" in system
        assert "SCOPE FIDELITY" in system
        assert "Do not PASS merely because every sentence in the candidate is true." in system

    def test_every_contract_is_registered(self):
        assert set(CONTRACTS) == {"v1", "v2", "v3", "v3_1"}


class TestV3_1IsExactlyV3PlusTwoInsertions:
    def test_removing_the_insertions_reproduces_v3(self):
        """The delta is provable, not asserted. One variable per comparison."""
        stripped = (
            CONTRACTS["v3_1"]
            .system.replace(_V3_1_MATERIALITY_BOUNDARY, "")
            .replace(_V3_1_DO_NOT_FAIL, "")
        )
        assert stripped == CONTRACTS["v3"].system

    def test_boundary_sits_immediately_after_material_completeness(self):
        system = CONTRACTS["v3_1"].system
        assert (
            system.index("MATERIAL COMPLETENESS")
            < system.index("MATERIALITY BOUNDARY")
            < system.index("SCOPE FIDELITY")
        )

    def test_do_not_fail_sits_beside_do_not_pass(self):
        """The counterweight has to be next to the instruction it balances."""
        system = CONTRACTS["v3_1"].system
        assert system.index("Do not PASS merely") < system.index("Do not FAIL a substantively")

    def test_template_is_unchanged_across_v2_v3_v3_1(self):
        assert CONTRACTS["v2"].template == CONTRACTS["v3"].template == CONTRACTS["v3_1"].template

    def test_v3_1_still_wants_source_context(self):
        assert CONTRACTS["v3_1"].wants_source_context


class TestTheBoundaryContent:
    def test_completeness_is_not_exhaustiveness(self):
        assert "Completeness does not mean exhaustiveness." in CONTRACTS["v3_1"].system

    def test_supporting_detail_is_named_as_non_material(self):
        system = CONTRACTS["v3_1"].system
        for phrase in ("examples", "dates", "procedural history", "monitoring methods",
                       "secondary quantitative figures"):
            assert phrase in system

    def test_the_test_question_is_stated(self):
        assert "would it only add supporting detail?" in CONTRACTS["v3_1"].system

    def test_smallest_central_set_is_the_grading_target(self):
        assert "smallest set of central claims" in CONTRACTS["v3_1"].system

    def test_scope_fidelity_is_untouched_by_the_boundary(self):
        """The boundary loosens completeness only. Scope must stay strict.

        CAT FX and P&G revenue were scope errors, not completeness errors --
        loosening scope would put them straight back.
        """
        system = CONTRACTS["v3_1"].system
        assert "Company-wide results must not be replaced by segment-level results" in system
        assert "Adjacent risks are not substitutes" in system
        assert "can still FAIL when those facts answer a narrower" in system


class TestKappaIsCountedPerPair:
    """24 items measured 3 times is not 72 observations."""

    def _runs(self, contract: str, verdicts: dict[str, list[bool]],
              human: dict[str, bool]) -> list[JudgeRun]:
        runs = []
        for pair_id, repeats in verdicts.items():
            for i, v in enumerate(repeats, start=1):
                runs.append(
                    JudgeRun(pair_id, "x", contract, "m", i, human[pair_id], verdict=v)
                )
        return runs

    def test_majority_decides_each_pair(self):
        runs = self._runs("v3_1", {"p": [True, True, False]}, {"p": True})
        cell = confusion_by_pair(runs)[0]
        assert cell.n == 1
        assert cell.tp == 1

    def test_n_is_the_number_of_pairs_not_repeats(self):
        runs = self._runs(
            "v3_1",
            {f"p{i}": [True, True, True] for i in range(24)},
            {f"p{i}": True for i in range(24)},
        )
        assert confusion_by_pair(runs)[0].n == 24

    def test_disagreeing_repeats_are_flagged_as_unstable(self):
        runs = self._runs("v3_1", {"p": [True, False, True]}, {"p": True})
        assert confusion_by_pair(runs)[0].unstable_pairs == ["p"]

    def test_unanimous_repeats_are_not_flagged(self):
        runs = self._runs("v3_1", {"p": [True, True, True]}, {"p": True})
        assert confusion_by_pair(runs)[0].unstable_pairs == []

    def test_per_pair_kappa_matches_the_hand_computed_v2_number(self):
        """12/0/7/5 -> 0.4167, the figure the human labels produced."""
        human, verdicts = {}, {}
        for i in range(12):
            human[f"tp{i}"], verdicts[f"tp{i}"] = True, [True] * 3
        for i in range(7):
            human[f"fp{i}"], verdicts[f"fp{i}"] = False, [True] * 3
        for i in range(5):
            human[f"tn{i}"], verdicts[f"tn{i}"] = False, [False] * 3
        cell = confusion_by_pair(self._runs("v2", verdicts, human))[0]
        assert cell.n == 24
        assert cell.kappa == pytest.approx(0.4167, abs=1e-4)


class TestAdoptionGate:
    """Declared before the run, checked mechanically, every clause reported."""

    def _grid(self, v3_1_verdicts: dict[str, bool]) -> list[JudgeRun]:
        human = {
            "nar-PG-fx-exposure": True,
            "nar-PG-legal": True,
            "nar-CAT-fx-exposure": False,
            "nar-AAPL-supply-concentration": False,
            "nar-COST-supply-concentration": False,
        }
        baseline = {
            "nar-PG-fx-exposure": True,
            "nar-PG-legal": True,
            "nar-CAT-fx-exposure": True,
            "nar-AAPL-supply-concentration": True,
            "nar-COST-supply-concentration": True,
        }
        runs = []
        for pair_id, expected in human.items():
            runs.append(JudgeRun(pair_id, "x", "v3", "m", 1, expected,
                                 verdict=baseline[pair_id]))
            runs.append(JudgeRun(pair_id, "x", "v3_1", "m", 1, expected,
                                 verdict=v3_1_verdicts[pair_id]))
        return runs

    def test_the_target_behaviour_is_adopted(self):
        result = check_adoption(
            self._grid(
                {
                    "nar-PG-fx-exposure": True,
                    "nar-PG-legal": True,
                    "nar-CAT-fx-exposure": False,
                    "nar-AAPL-supply-concentration": False,
                    "nar-COST-supply-concentration": False,
                }
            ),
            candidate="v3_1",
            baseline="v3",
            model="m",
        )
        assert result["adopted"], result["failed_clauses"]

    def test_a_named_must_pass_case_failing_blocks_adoption(self):
        result = check_adoption(
            self._grid(
                {
                    "nar-PG-fx-exposure": False,   # the v3 regression, unfixed
                    "nar-PG-legal": True,
                    "nar-CAT-fx-exposure": False,
                    "nar-AAPL-supply-concentration": False,
                    "nar-COST-supply-concentration": False,
                }
            ),
            candidate="v3_1",
            baseline="v3",
            model="m",
        )
        assert not result["adopted"]
        assert any("nar-PG-fx-exposure = PASS" in c for c in result["failed_clauses"])

    def test_losing_a_must_fail_case_blocks_adoption(self):
        """Passing P&G again must not be bought by going permissive on Costco."""
        result = check_adoption(
            self._grid(
                {
                    "nar-PG-fx-exposure": True,
                    "nar-PG-legal": True,
                    "nar-CAT-fx-exposure": False,
                    "nar-AAPL-supply-concentration": False,
                    "nar-COST-supply-concentration": True,   # regressed to PASS
                }
            ),
            candidate="v3_1",
            baseline="v3",
            model="m",
        )
        assert not result["adopted"]
        assert any("nar-COST-supply-concentration = FAIL" in c for c in result["failed_clauses"])

    def test_every_clause_is_reported_whether_it_passed_or_not(self):
        result = check_adoption(self._grid(
            {k: v for k, v in zip(
                ["nar-PG-fx-exposure", "nar-PG-legal", "nar-CAT-fx-exposure",
                 "nar-AAPL-supply-concentration", "nar-COST-supply-concentration"],
                [True, True, False, False, False])}
        ), candidate="v3_1", baseline="v3", model="m")
        names = [c["clause"] for c in result["clauses"]]
        assert "new false negatives = 0" in names
        assert "FP <= 4" in names
        assert any("kappa >=" in n for n in names)
        assert len(names) == 8

    def test_kappa_alone_cannot_carry_adoption(self):
        """The v3 situation: threshold cleared, correct answers broken."""
        result = check_adoption(
            self._grid(
                {
                    "nar-PG-fx-exposure": False,
                    "nar-PG-legal": False,
                    "nar-CAT-fx-exposure": False,
                    "nar-AAPL-supply-concentration": False,
                    "nar-COST-supply-concentration": False,
                }
            ),
            candidate="v3_1",
            baseline="v3",
            model="m",
        )
        assert not result["adopted"]
        assert "new false negatives = 0" in result["failed_clauses"]

    def test_an_ungraded_case_is_not_silently_treated_as_passing(self):
        runs = [JudgeRun("nar-PG-legal", "x", "v3_1", "m", 1, True, verdict=True)]
        result = check_adoption(runs, candidate="v3_1", baseline="v3", model="m")
        missing = [c for c in result["clauses"] if c["detail"] == "not graded"]
        assert missing
        assert not result["adopted"]


class TestDefaultStaysPut:
    def test_v3_1_is_not_the_default_before_it_is_measured(self):
        from edgar_intel.evals.judge_contracts import DEFAULT_CONTRACT

        assert DEFAULT_CONTRACT != "v3_1"
        assert get_contract().name == DEFAULT_CONTRACT
