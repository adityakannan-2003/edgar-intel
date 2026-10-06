"""The CI eval gate compares against a committed baseline.

CI's database starts empty, so `eval gate` never found a run labelled
"baseline" there and passed having compared nothing. With `--baseline-file` it
compares against a summary committed beside the fixture, and every way that
comparison could be meaningless fails instead of passing: a missing or
malformed file, a golden set that is not the exam the baseline measured, or a
run on only part of it.

No database: run summaries are handed to the gate through `latest_run`.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from edgar_intel.evals import report
from edgar_intel.evals.goldenset import save
from edgar_intel.evals.schemas import EvalCase


def case(cid: str, expected: str = "$1.00 billion", value: float = 1e9, labels=("1",)):
    return EvalCase(
        case_id=cid,
        kind="numeric",
        question=f"What was {cid}?",
        expected=expected,
        expected_value=value,
        unit="USD",
        relevant_chunk_ids=list(labels),
    )


CASES = [case("num-ALFA-Assets-2023"), case("num-BETA-Assets-2023", "$2.00 billion", 2e9)]

SUMMARY = {
    "n_cases": 2,
    "numeric_accuracy": 0.5,
    "narrative_pass_rate": 0.0,
    "overall_score": 0.5,
    "abstention_rate": 0.0,
    "hallucination_rate": 0.5,
    "retrieval": {"recall@5": 1.0, "hit@5": 1.0, "retrieve_ms": 2.0},
    "p95_latency_ms": 40,
    "config": {"strategy": "section_aware"},
}


def run(summary: dict, run_key: str = "ci-head") -> dict:
    return {"id": 1, "run_key": run_key, "label": "ci", "git_sha": "abc", "summary": json.dumps(summary)}


def with_recall(recall: float) -> dict:
    return {**SUMMARY, "retrieval": {**SUMMARY["retrieval"], "recall@5": recall}}


@pytest.fixture
def golden(tmp_path) -> str:
    path = tmp_path / "golden.json"
    save(CASES, str(path))
    return str(path)


@pytest.fixture
def latest(monkeypatch):
    """Install the current run, and optionally a database baseline run."""

    def install(current: dict, baseline: dict | None = None) -> None:
        monkeypatch.setattr(report, "latest_run", lambda label=None: baseline if label else current)

    return install


@pytest.fixture
def baseline_file(tmp_path, golden, monkeypatch):
    """A baseline written by `save_baseline` from a run on `golden`."""

    def write(summary: dict = SUMMARY) -> str:
        out = tmp_path / "ci_baseline.json"
        monkeypatch.setattr(report.db, "query_one", lambda sql, params: run(summary, "ci-base"))
        report.save_baseline("ci-base", golden, str(out))
        return str(out)

    return write


class TestWithoutABaselineFile:
    def test_an_empty_database_still_passes_and_says_so(self, latest):
        """Unchanged: the database mode is what local runs use."""
        latest(run(SUMMARY))
        result = report.gate()
        assert result.passed
        assert "no prior run labelled 'baseline'" in result.reasons[0]


class TestAgainstABaselineFile:
    def test_the_run_it_was_saved_from_passes(self, latest, baseline_file, golden):
        path = baseline_file()
        latest(run(SUMMARY))
        result = report.gate(baseline_file=path, golden_path=golden)
        assert result.passed, result.render()
        assert result.deltas == {"overall_score": 0.0, "numeric_accuracy": 0.0, "recall@5": 0.0}

    def test_a_regression_beyond_tolerance_fails(self, latest, baseline_file, golden):
        path = baseline_file()
        latest(run(with_recall(0.9)))
        result = report.gate(baseline_file=path, golden_path=golden)
        assert not result.passed
        assert "recall@5 regressed by 0.1000 (1.0000 -> 0.9000)" in result.render()

    def test_a_drop_within_tolerance_passes(self, latest, baseline_file, golden):
        path = baseline_file()
        latest(run(with_recall(0.98)))
        assert report.gate(baseline_file=path, golden_path=golden).passed

    def test_an_improvement_passes(self, latest, baseline_file, golden):
        path = baseline_file()
        latest(run({**SUMMARY, "numeric_accuracy": 1.0, "overall_score": 1.0}))
        result = report.gate(baseline_file=path, golden_path=golden)
        assert result.passed
        assert result.deltas["numeric_accuracy"] == 0.5

    def test_a_missing_file_fails(self, latest, golden, tmp_path):
        """The silent pass this mode exists to remove."""
        latest(run(SUMMARY))
        result = report.gate(baseline_file=str(tmp_path / "absent.json"), golden_path=golden)
        assert not result.passed
        assert "is unusable" in result.reasons[0]

    def test_a_malformed_file_fails(self, latest, golden, tmp_path):
        path = tmp_path / "ci_baseline.json"
        path.write_text("{}")
        latest(run(SUMMARY))
        assert not report.gate(baseline_file=str(path), golden_path=golden).passed

    def test_another_exam_fails(self, latest, baseline_file, golden):
        """A changed expected answer is a new exam: comparing would mean nothing."""
        path = baseline_file()
        save([CASES[0], case("num-BETA-Assets-2023", "$2.10 billion", 2.1e9)], golden)
        latest(run(SUMMARY))
        result = report.gate(baseline_file=path, golden_path=golden)
        assert not result.passed
        assert "is not the exam the baseline was measured on" in result.reasons[0]

    def test_a_run_on_part_of_the_exam_fails(self, latest, baseline_file, golden):
        path = baseline_file()
        latest(run({**SUMMARY, "n_cases": 1}))
        result = report.gate(baseline_file=path, golden_path=golden)
        assert not result.passed
        assert "the run graded 1 cases" in result.reasons[0]


class TestTheFingerprint:
    def test_evidence_labels_do_not_change_it(self):
        """Labels are chunk ids of whichever index the set was linked against."""
        relabelled = [case(c.case_id, c.expected, c.expected_value, labels=("99",)) for c in CASES]
        assert report.golden_fingerprint(relabelled) == report.golden_fingerprint(CASES)

    def test_order_does_not_change_it(self):
        assert report.golden_fingerprint(CASES[::-1]) == report.golden_fingerprint(CASES)

    def test_an_expected_value_does(self):
        changed = [CASES[0], case("num-BETA-Assets-2023", "$2.00 billion", 2.01e9)]
        assert report.golden_fingerprint(changed) != report.golden_fingerprint(CASES)


class TestSaveBaseline:
    def test_keeps_what_the_gate_compares_and_drops_what_the_machine_decides(
        self, baseline_file
    ):
        saved = json.loads(pathlib.Path(baseline_file()).read_text())
        assert saved["golden"]["n_cases"] == 2
        assert saved["summary"]["numeric_accuracy"] == 0.5
        assert saved["summary"]["retrieval"] == {"recall@5": 1.0, "hit@5": 1.0}
        assert "p95_latency_ms" not in saved["summary"]
        assert "config" not in saved["summary"]

    def test_refuses_a_run_on_a_different_number_of_cases(self, baseline_file):
        with pytest.raises(ValueError, match="graded 3 cases"):
            baseline_file({**SUMMARY, "n_cases": 3})


class TestTheCommittedBaseline:
    def test_it_gates_the_three_metrics_on_the_fixture_exam(self):
        saved = json.loads(pathlib.Path("tests/fixtures/ci_baseline.json").read_text())
        assert saved["golden"]["n_cases"] == saved["summary"]["n_cases"] == 24
        assert len(saved["golden"]["sha256"]) == 16
        for path, _ in report.GATED_METRICS:
            assert report._extract(saved["summary"], path) is not None, path
        assert not any(k.endswith("_ms") for k in saved["summary"]["retrieval"])
