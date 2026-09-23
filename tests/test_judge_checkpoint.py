"""The calibration checkpoint: repeats, duplicate controls, prompt structure.

Two rubric iterations produced no measurable change, and the reason turned out to
be underneath the question: two nominally identical context-free runs of v2 on
the same 24 labels gave kappa 0.4167 (FP 7, FN 0) and 0.5833 (FP 4, FN 1). The
judge disagreeing with itself was as large as every contract difference measured.

So this checkpoint is about the instrument, not the rubric -- more repeats, more
labels, a measured noise floor, and one structural arm. The tests below exist to
keep each of those from quietly reverting.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from edgar_intel.evals.judge_contracts import (
    CONTRACTS,
    DEFAULT_STRUCTURE,
    STRUCTURES,
    get_structure,
)
from edgar_intel.evals.judge_lab import (
    CALIBRATION_REPEATS,
    JudgeRun,
    beats_noise,
    check_adoption,
    composition,
    confusion_by_pair,
    duplicate_variance,
    flips,
    run_grid,
)
from edgar_intel.evals.judge_lab import FrozenPair


def pair(pid: str, human: bool = True, context: str = "ctx") -> FrozenPair:
    return FrozenPair(
        pair_id=pid,
        question="q",
        reference="ref",
        answer="ans",
        source_context=context,
        expected_verdict=human,
        label="human-pass" if human else "human-fail",
    )


class TestCalibrationRepeats:
    def test_the_constant_is_nine(self):
        assert CALIBRATION_REPEATS == 9

    def test_the_cli_default_matches_the_constant(self):
        """typer is not installed here, so read the default off the parse tree.

        Pinning both places together is the point: the literal in cli.py and the
        constant in judge_lab are allowed to exist separately but not to drift.
        """
        source = pathlib.Path("src/edgar_intel/cli.py").read_text()
        tree = ast.parse(source)
        fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "eval_judge_kappa"
        )
        names = [a.arg for a in fn.args.args]
        default = fn.args.defaults[names.index("repeats") - (len(names) - len(fn.args.defaults))]
        # typer.Option(9, help=...) -> first positional argument
        assert isinstance(default, ast.Call)
        assert default.args[0].value == CALIBRATION_REPEATS

    def test_no_stale_seventy_two_observation_claim(self):
        """The old comment hard-coded 24 pairs x 3 repeats. Both numbers moved."""
        for path in ("src/edgar_intel/evals/judge_lab.py", "src/edgar_intel/cli.py"):
            text = pathlib.Path(path).read_text()
            assert "72 independent" not in text, path
            assert "24 items measured 3 times" not in text, path


class TestPromptStructures:
    def test_current_is_the_default_and_the_control(self):
        assert DEFAULT_STRUCTURE == "current"
        assert get_structure().name == "current"
        assert get_structure("current").template == CONTRACTS["v2"].template

    def test_reference_last_puts_the_reference_after_the_candidate(self):
        template = STRUCTURES["reference_last"].template
        assert (
            template.index("SOURCE CONTEXT")
            < template.index("CANDIDATE ANSWER")
            < template.index("REFERENCE ANSWER")
        )

    def test_reference_last_carries_the_coverage_standard(self):
        system_text = STRUCTURES["reference_last"].template
        assert "defines the required coverage standard" in system_text
        assert "Do not treat factual support from the source context as sufficient" in system_text

    def test_a_layout_only_arm_exists_to_decompose_the_bundle(self):
        """reference_last changes order AND adds a sentence. That is two things."""
        layout_only = STRUCTURES["reference_last_layout_only"].template
        assert "REFERENCE ANSWER" in layout_only
        assert "defines the required coverage standard" not in layout_only

    def test_structures_do_not_touch_the_contract_system_prompt(self):
        """A layout comparison is only a layout comparison if the rubric is fixed."""
        before = CONTRACTS["v3_1"].system
        for name in STRUCTURES:
            get_structure(name).render("q", "ref", "cand", "ctx")
        assert CONTRACTS["v3_1"].system == before

    def test_every_structure_fills_every_slot(self):
        for name, structure in STRUCTURES.items():
            rendered = structure.render("QQ", "RR", "CC", "XX")
            for token in ("QQ", "RR", "CC", "XX"):
                assert token in rendered, f"{name} dropped {token}"

    def test_missing_context_is_still_marked(self):
        assert "(not supplied)" in get_structure("reference_last").render("q", "r", "c", "")

    def test_unknown_structure_raises(self):
        with pytest.raises(ValueError):
            get_structure("sideways")

    def test_judge_narrative_takes_structure_separately_from_contract(self):
        from edgar_intel.evals.judge import judge_narrative

        params = inspect.signature(judge_narrative).parameters
        for knob in ("contract", "structure", "model", "source_context"):
            assert knob in params


class TestDuplicateControl:
    def _runs(self, structure: str, replicate: int, verdicts: dict[str, bool]) -> list[JudgeRun]:
        return [
            JudgeRun(
                pair_id=pid,
                label="x",
                contract="v3_1",
                model="m",
                repeat=1,
                expected_verdict=(pid.startswith("good")),
                structure=structure,
                replicate=replicate,
                context_supplied=True,
                verdict=v,
            )
            for pid, v in verdicts.items()
        ]

    def test_duplicates_are_separate_cells_not_averaged(self):
        runs = self._runs("current", 1, {"good1": True, "bad1": True}) + self._runs(
            "current", 2, {"good1": True, "bad1": False}
        )
        cells = confusion_by_pair(runs)
        assert len(cells) == 2
        assert {c.replicate for c in cells} == {1, 2}

    def test_variance_reports_the_spread_and_the_flipped_pairs(self):
        runs = self._runs("current", 1, {"good1": True, "bad1": True}) + self._runs(
            "current", 2, {"good1": True, "bad1": False}
        )
        rows = duplicate_variance(confusion_by_pair(runs))
        assert len(rows) == 1
        assert rows[0]["verdicts_flipped"] == 1
        assert "bad1" in rows[0]["flipped_pairs"]
        assert rows[0]["d_kappa"] > 0

    def test_no_duplicate_means_no_variance_row(self):
        rows = duplicate_variance(confusion_by_pair(self._runs("current", 1, {"good1": True})))
        assert rows == []

    def test_cell_id_distinguishes_replicates(self):
        cells = confusion_by_pair(
            self._runs("current", 1, {"good1": True}) + self._runs("current", 2, {"good1": True})
        )
        assert cells[0].cell_id != cells[1].cell_id

    def test_grid_builds_a_duplicate_cell(self):
        """No model call: an unknown contract raises before any judging."""
        with pytest.raises(ValueError, match="cannot duplicate"):
            run_grid([pair("p")], contracts=("v2",), structures=("current",),
                     duplicate="v3_1/reference_last", repeats=1)

    def test_grid_rejects_an_unknown_contract_before_spending_anything(self):
        with pytest.raises(ValueError, match="unknown contract"):
            run_grid([pair("p")], contracts=("v99",), repeats=1)


class TestNoiseFloorGatesTheConclusion:
    VARIANCE = [{"condition": "v3_1/current/m", "d_kappa": 0.0833, "verdicts_flipped": 2}]

    def test_an_effect_inside_the_spread_is_unresolved(self):
        ok, message = beats_noise(0.0417, self.VARIANCE)
        assert not ok
        assert "UNRESOLVED" in message

    def test_an_effect_larger_than_the_spread_counts(self):
        ok, message = beats_noise(0.2500, self.VARIANCE)
        assert ok
        assert "exceeds" in message

    def test_sign_does_not_matter_only_magnitude(self):
        assert beats_noise(-0.25, self.VARIANCE)[0]

    def test_without_a_duplicate_no_delta_can_be_called_an_effect(self):
        ok, message = beats_noise(0.9, [])
        assert not ok
        assert "no measured noise floor" in message

    def test_the_gate_gains_a_noise_clause_when_variance_is_supplied(self):
        runs = [
            JudgeRun("good1", "x", "v3_1", "m", 1, True, structure="current", verdict=True),
            JudgeRun("good1", "x", "v3_1", "m", 1, True, structure="reference_last", verdict=True),
        ]
        result = check_adoption(
            runs,
            candidate="v3_1/reference_last",
            baseline="v3_1/current",
            model="m",
            variance_rows=self.VARIANCE,
        )
        clauses = [c["clause"] for c in result["clauses"]]
        assert "effect exceeds duplicate-run noise" in clauses

    def test_the_gate_omits_the_clause_when_no_duplicate_ran(self):
        runs = [JudgeRun("good1", "x", "v3_1", "m", 1, True, structure="current", verdict=True)]
        result = check_adoption(runs, candidate="v3_1", baseline="v3_1", model="m")
        clauses = [c["clause"] for c in result["clauses"]]
        assert "effect exceeds duplicate-run noise" not in clauses


class TestFlipsAcrossStructures:
    def _runs(self) -> list[JudgeRun]:
        out = []
        for structure, verdicts in (
            ("current", {"nar-PG-legal": False, "nar-CAT-fx-exposure": True}),
            ("reference_last", {"nar-PG-legal": True, "nar-CAT-fx-exposure": False}),
        ):
            for pid, v in verdicts.items():
                out.append(
                    JudgeRun(
                        pid, "x", "v3_1", "m", 1,
                        expected_verdict=(pid == "nar-PG-legal"),
                        structure=structure, verdict=v,
                    )
                )
        return out

    def test_a_structure_comparison_finds_its_flips(self):
        rows = flips(self._runs(), "v3_1/current", "v3_1/reference_last", "m")
        assert len(rows) == 2
        effects = {r["pair_id"]: r["effect"] for r in rows}
        assert "false negative removed" in effects["nar-PG-legal"]
        assert "false positive removed" in effects["nar-CAT-fx-exposure"]

    def test_a_bare_contract_still_resolves_for_older_call_sites(self):
        runs = [
            JudgeRun("p", "x", "v2", "m", 1, False, structure="current", verdict=True),
            JudgeRun("p", "x", "v3", "m", 1, False, structure="current", verdict=False),
        ]
        rows = flips(runs, "v2", "v3", "m")
        assert len(rows) == 1


class TestSetComposition:
    def test_counts_by_kind_ticker_and_label(self):
        pairs = [
            pair("nar-AAPL-legal", True),
            pair("nar-AAPL-supply-concentration", False),
            pair("nar-PG-fx-exposure", True),
            pair("nar-CAT-fx-exposure", False),
        ]
        comp = composition(pairs)
        assert comp["total"] == 4
        assert comp["human_pass"] == 2
        assert comp["human_fail"] == 2
        assert comp["by_kind"]["fx-exposure"] == 2
        assert comp["by_ticker"]["AAPL"] == 2

    def test_a_small_set_is_flagged(self):
        assert any("40 is the floor" in w for w in composition([pair("nar-AAPL-legal")])["warnings"])

    def test_a_set_dominated_by_one_kind_is_flagged(self):
        pairs = [pair(f"nar-T{i}-legal", i % 2 == 0) for i in range(10)]
        assert any("question type" in w for w in composition(pairs)["warnings"])

    def test_a_lopsided_label_split_is_flagged(self):
        """A 9:1 set inflates chance agreement and caps the kappa on offer."""
        kinds = ["legal", "fx-exposure", "supply", "revenue-drivers", "competition"]
        pairs = [pair(f"nar-T{i}-{kinds[i % 5]}", True) for i in range(9)]
        pairs.append(pair("nar-T9-legal", False))
        assert any("lopsided" in w for w in composition(pairs)["warnings"])

    def test_a_balanced_forty_pair_set_raises_nothing(self):
        kinds = ["legal", "fx-exposure", "supply", "revenue-drivers", "competition"]
        tickers = ["AAPL", "MSFT", "NVDA", "COST", "JNJ", "CAT", "UNH", "PG"]
        pairs = [
            pair(f"nar-{tickers[i % 8]}-{kinds[i % 5]}-{i}", i % 2 == 0) for i in range(40)
        ]
        assert composition(pairs)["warnings"] == []


class TestEmptyNumericPopulation:
    """A real zero and an empty set are different facts."""

    def _summary(self, results):
        from edgar_intel.evals.runner import summarise_run

        return summarise_run("k", "l", "sha", results, {})

    def _narrative(self, passed: bool):
        from edgar_intel.evals.schemas import CaseResult

        return CaseResult("nar-1", "narrative", passed, 1.0 if passed else 0.0, "a", "e")

    def _numeric(self, passed: bool):
        from edgar_intel.evals.schemas import CaseResult

        return CaseResult("num-1", "numeric", passed, 1.0 if passed else 0.0, "a", "e")

    def test_no_numeric_cases_gives_null_not_zero(self):
        summary = self._summary([self._narrative(True)])
        assert summary.numeric_accuracy is None

    def test_numeric_cases_that_all_fail_still_give_zero(self):
        summary = self._summary([self._numeric(False)])
        assert summary.numeric_accuracy == 0.0

    def test_overall_score_ignores_an_absent_numeric_population(self):
        summary = self._summary([self._narrative(True), self._narrative(False)])
        assert summary.overall_score == 0.5

    def test_a_mixed_run_is_unaffected(self):
        summary = self._summary([self._numeric(True), self._narrative(False)])
        assert summary.numeric_accuracy == 1.0


class TestKappaBelongsToTheCalibrationSet:
    def test_no_overlap_is_stated_as_a_category_not_a_todo(self):
        from edgar_intel.evals.judge import kappa_verdict

        message = kappa_verdict(None, n_labels_matched=0)
        assert "not applicable to this run" in message
        assert "frozen calibration set" in message
        assert "judge-kappa" in message

    def test_partial_overlap_below_the_floor_says_so(self):
        from edgar_intel.evals.judge import kappa_verdict

        assert "only 7" in kappa_verdict(None, n_labels_matched=7)

    def test_a_measured_kappa_is_reported_normally(self):
        from edgar_intel.evals.judge import kappa_verdict

        assert "0.75" in kappa_verdict(0.75)

    def test_the_summary_records_how_many_labels_matched(self):
        from edgar_intel.evals.schemas import RunSummary

        assert "judge_labels_matched" in RunSummary.__dataclass_fields__


class TestMigrationNumbering:
    def test_no_two_migrations_share_a_number(self):
        numbers = [p.name.split("_", 1)[0] for p in pathlib.Path("sql").glob("*.sql")]
        assert len(numbers) == len(set(numbers)), sorted(numbers)

    def test_migrations_sort_into_a_single_order(self):
        names = sorted(p.name for p in pathlib.Path("sql").glob("*.sql"))
        assert names == ["001_schema.sql", "002_fact_period_uniqueness.sql",
                         "003_eval_result_replay_context.sql"]
