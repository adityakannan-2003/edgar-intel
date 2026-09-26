"""`eval regrade`: measure a grader fix on stored answers, without a paid run.

D2 changes verdicts without changing a single answer. Re-running the suite to
measure it would also regenerate every answer, mixing provider drift into the
delta; re-marking the answers already in `eval_results` isolates the grader.

Two refusals matter more than the arithmetic: a case whose expected answer
changed since the run is skipped rather than re-graded (a delta across a
golden-set rebuild compares two different exams), and the run's own numeric
tolerance is used so the grader code is the only variable.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from edgar_intel.evals.regrade import regrade_rows
from edgar_intel.evals.schemas import EvalCase

D2_ANSWER = (
    "Caterpillar Inc.'s research and development expense for FY2024 was $2,107 "
    "million, which is a decrease from $2,108 million in FY2023."
)


def _cases() -> dict[str, EvalCase]:
    return {
        "yoy-CAT-RD": EvalCase(
            case_id="yoy-CAT-RD", kind="numeric", question="How did it change?",
            expected="decreased 0.0%", expected_value=-0.0474, unit="percent",
            difficulty="comparative", notes="FY2023=2,108,000,000, FY2024=2,107,000,000",
        ),
        "num-CAT-Rev": EvalCase(
            case_id="num-CAT-Rev", kind="numeric", question="Revenue?",
            expected="$64.81 billion", expected_value=64_809_000_000, unit="USD",
        ),
        "num-CAT-Cash": EvalCase(
            case_id="num-CAT-Cash", kind="numeric", question="Cash?",
            expected="$6.89 billion", expected_value=6_889_000_000, unit="USD",
        ),
        "num-CAT-Debt": EvalCase(
            case_id="num-CAT-Debt", kind="numeric", question="Debt?",
            expected="$38.0 billion", expected_value=38_000_000_000, unit="USD",
        ),
    }


def _rows() -> list[dict]:
    return [
        {"case_id": "yoy-CAT-RD", "passed": False, "answer": D2_ANSWER,
         "expected": "decreased 0.0%", "judge_rationale": "no percentage found in the answer"},
        {"case_id": "num-CAT-Rev", "passed": True, "answer": "Revenue was $64,809 million.",
         "expected": "$64.81 billion", "judge_rationale": "match at millions scale"},
        {"case_id": "num-CAT-Cash", "passed": False,
         "answer": "The context does not provide the cash balance.",
         "expected": "$6.89 billion", "judge_rationale": "ABSTAINED"},
        {"case_id": "num-CAT-Debt", "passed": False, "answer": "",
         "expected": "$38.0 billion", "judge_rationale": "401 Unauthorized"},
        {"case_id": "num-GONE", "passed": False, "answer": "$1", "expected": "x",
         "judge_rationale": ""},
    ]


class TestRegradeRows:
    def test_the_d2_answer_flips_and_nothing_else_moves(self):
        out = regrade_rows(_rows(), _cases(), tolerance=0.005)
        assert [f["case_id"] for f in out["fail_to_pass"]] == ["yoy-CAT-RD"]
        assert out["pass_to_fail"] == []
        assert out["before"]["passed"] == 1 and out["after"]["passed"] == 2

    def test_rates_use_the_runs_own_population(self):
        """Errored rows are excluded, as summarise_run excludes them."""
        out = regrade_rows(_rows(), _cases(), tolerance=0.005)
        assert out["numeric_graded"] == 3
        assert out["numeric_errored"] == 1
        assert out["before"]["numeric_accuracy"] == round(1 / 3, 4)
        assert out["after"]["numeric_accuracy"] == round(2 / 3, 4)

    def test_a_false_failure_was_being_counted_as_a_hallucination(self):
        """The D2 answers sat in the hallucination bucket, not the abstention one."""
        out = regrade_rows(_rows(), _cases(), tolerance=0.005)
        assert out["before"]["wrong"] == 1 and out["after"]["wrong"] == 0
        assert out["before"]["abstained"] == out["after"]["abstained"] == 1

    def test_a_changed_expectation_is_skipped_not_regraded(self):
        rows = _rows()
        rows[1]["expected"] = "$64.80 billion"  # the golden set moved since the run
        out = regrade_rows(rows, _cases(), tolerance=0.005)
        assert out["skipped"]["expected_changed"] == 1
        assert out["skipped"]["not_in_golden_set"] == 1

    def test_a_verdict_that_gets_worse_is_reported(self):
        rows = _rows()
        rows[1]["answer"] = "Revenue was $70,000 million."  # stored as a pass
        out = regrade_rows(rows, _cases(), tolerance=0.005)
        assert [f["case_id"] for f in out["pass_to_fail"]] == ["num-CAT-Rev"]

    def test_by_difficulty(self):
        out = regrade_rows(_rows(), _cases(), tolerance=0.005)
        assert out["by_difficulty"]["comparative"] == {"cases": 1, "before": 0, "after": 1}


class TestFailureAttribution:
    """"62% of failures are retrieval misses" is a headline claim, and D2's false
    failures sat in the other bucket. The re-marked run is attributed by the
    same rule `eval failures` used for the original."""

    def test_the_rule_is_shared_with_eval_failures(self):
        from edgar_intel.evals.report import attribute_failure

        assert attribute_failure({"recall@5": 0.0}) == "retrieval"
        assert attribute_failure('{"recall@5": 0.4}') == "generation"
        assert attribute_failure({}) == "unlabelled"
        assert attribute_failure(None) == "unlabelled"

    def test_a_flipped_generation_miss_leaves_the_generation_bucket(self):
        rows = _rows()
        rows[0]["retrieval"] = {"recall@5": 0.5}   # the D2 answer had its evidence
        rows[2]["retrieval"] = {"recall@5": 0.0}   # the abstention did not
        out = regrade_rows(rows, _cases(), tolerance=0.005)
        before, after = out["failure_attribution"]["before"], out["failure_attribution"]["after"]
        assert before["generation"] == 1 and after["generation"] == 0
        assert before["retrieval"] == after["retrieval"] == 1

    def test_non_numeric_rows_keep_their_verdict_but_are_attributed(self):
        rows = _rows() + [
            {"case_id": "nar-AAPL-legal", "kind": "narrative", "passed": False,
             "answer": "various proceedings", "expected": "ref",
             "retrieval": {"recall@5": 0.0}},
        ]
        out = regrade_rows(rows, _cases(), tolerance=0.005)
        assert out["failure_attribution"]["after"]["retrieval"] == 1
        assert out["numeric_graded"] == 3  # the narrative row is not re-marked


class TestRegradeRun:
    @staticmethod
    def _db(monkeypatch, config: dict, summary: dict, rows: list[dict] | None = None):
        import edgar_intel.db as db

        monkeypatch.setattr(
            db, "query_one",
            lambda sql, params=None: {
                "id": 9, "run_key": "baseline-v5-x", "git_sha": "a9e9c08",
                "config": config, "summary": summary,
            },
        )
        monkeypatch.setattr(db, "query", lambda sql, params=None: rows or _rows())
        monkeypatch.setattr(
            db, "execute", lambda *a, **k: pytest.fail("regrade must not write to the database")
        )

    def test_it_writes_a_report_and_never_the_database(self, tmp_path, monkeypatch):
        from edgar_intel.evals.regrade import regrade_run

        self._db(monkeypatch, {"numeric_tolerance": 0.005}, {"numeric_accuracy": 0.3333})
        out = tmp_path / "r.json"
        payload = regrade_run("baseline-v5-x", list(_cases().values()), "g.json", str(out))
        saved = json.loads(out.read_text())
        assert saved["after"]["numeric_accuracy"] == payload["after"]["numeric_accuracy"]
        assert saved["run_git_sha"] == "a9e9c08"
        assert saved["stored_verdicts_reproduce_report"] is True

    def test_the_runs_tolerance_is_the_one_used(self, tmp_path, monkeypatch):
        """$64,850M is 0.06% from $64,809M: a pass at the default 0.5%, a fail at
        the zero tolerance this run recorded. Only the run's own value is fair."""
        from edgar_intel.evals.regrade import regrade_run

        rows = _rows()
        rows[1]["answer"] = "Revenue was $64,850 million."
        self._db(monkeypatch, {"numeric_tolerance": 0.0}, {"numeric_accuracy": 0.3333}, rows)
        payload = regrade_run(
            "baseline-v5-x", list(_cases().values()), out_path=str(tmp_path / "r.json")
        )
        assert payload["numeric_tolerance"] == 0.0
        assert [f["case_id"] for f in payload["pass_to_fail"]] == ["num-CAT-Rev"]

    def test_stored_verdicts_that_do_not_reproduce_the_report_are_flagged(
        self, tmp_path, monkeypatch
    ):
        from edgar_intel.evals.regrade import regrade_run

        self._db(monkeypatch, {"numeric_tolerance": 0.005}, {"numeric_accuracy": 0.6202})
        payload = regrade_run(
            "baseline-v5-x", list(_cases().values()), out_path=str(tmp_path / "r.json")
        )
        assert payload["stored_verdicts_reproduce_report"] is False

    def test_it_refuses_to_overwrite(self, tmp_path, monkeypatch):
        from edgar_intel.evals.regrade import regrade_run

        out = tmp_path / "r.json"
        out.write_text("{}")
        with pytest.raises(FileExistsError):
            regrade_run("baseline-v5-x", [], out_path=str(out))


def test_it_is_wired_as_a_command():
    cli = pathlib.Path("src/edgar_intel/cli.py").read_text()
    assert '@eval_app.command("regrade")' in cli
    assert "regrade_run(" in cli
