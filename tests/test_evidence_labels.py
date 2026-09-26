"""Relevance labels are chunk ids from one index build (D11).

The golden set stores `relevant_chunk_ids` linked against section_aware on the
day it was built. Two operations on the roadmap would have graded against ids
that do not exist in the index being measured, with no error:

- `eval compare`: a `fixed` chunk id never equals a `section_aware` one, so the
  other three strategies score hit@k = 0 by construction.
- the D8 re-chunk: `index build` deletes and re-inserts, BIGSERIAL never reuses
  an id, and the next run reports hit@5 = 0.0 -- a number describing missing
  labels rather than retrieval, which is the void-run defect in a new place.

No database here, so the SQL is faked and the wiring is checked structurally,
as the other plumbing tests in this repo do.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from edgar_intel.evals.schemas import EvalCase


def _case(case_id: str, ids: list[str]) -> EvalCase:
    return EvalCase(
        case_id=case_id, kind="numeric", question="q?", expected="1",
        expected_value=1.0, relevant_chunk_ids=list(ids), ticker="CAT", fiscal_year=2024,
    )


def _fake_index(monkeypatch, present: dict[str, set[str]]) -> list:
    """`present` maps strategy -> chunk ids that exist under it."""
    import edgar_intel.db as db

    calls: list = []

    def query(sql, params=None):
        calls.append((sql, params))
        strategy, ids = params
        return [{"id": str(i)} for i in ids if str(i) in present.get(strategy, set())]

    monkeypatch.setattr(db, "query", query)
    return calls


def _fake_linker(monkeypatch, new_ids: dict[str, list[str]]) -> list:
    import edgar_intel.evals.goldenset as goldenset

    seen: list = []

    def link_evidence(cases, strategy, k=5):
        seen.append(strategy)
        for c in cases:
            c.relevant_chunk_ids = list(new_ids.get(c.case_id, []))
        return cases

    monkeypatch.setattr(goldenset, "link_evidence", link_evidence)
    return seen


class TestLabelCheck:
    def test_counts_labels_present_under_this_strategy_only(self, monkeypatch):
        from edgar_intel.evals.evidence import label_check

        calls = _fake_index(monkeypatch, {"section_aware": {"10", "11"}})
        cases = [_case("a", ["10", "11"]), _case("b", ["12"]), _case("c", [])]
        record = label_check(cases, "section_aware")
        assert record == {
            "strategy": "section_aware", "cases_labelled": 2, "labels": 3, "labels_in_index": 2,
        }
        sql, params = calls[0]
        assert "strategy = %s" in sql
        assert params == ("section_aware", [10, 11, 12])

    def test_no_labels_means_no_query(self, monkeypatch):
        from edgar_intel.evals.evidence import label_check

        calls = _fake_index(monkeypatch, {})
        assert label_check([_case("a", [])], "fixed")["labels"] == 0
        assert calls == []


class TestLabelsFor:
    def test_valid_labels_are_returned_untouched(self, monkeypatch):
        from edgar_intel.evals.evidence import labels_for

        _fake_index(monkeypatch, {"section_aware": {"10", "11"}})
        linked = _fake_linker(monkeypatch, {})
        cases = [_case("a", ["10"]), _case("b", ["11"])]
        out, record = labels_for(cases, "section_aware")
        assert out is cases
        assert record["relinked"] is False and record["stale_labels"] == 0
        assert linked == []

    def test_after_a_rechunk_every_label_is_stale_and_is_rederived(self, monkeypatch):
        """BIGSERIAL never reuses an id: a rebuilt index contains none of them."""
        from edgar_intel.evals.evidence import labels_for

        _fake_index(monkeypatch, {"section_aware": {"900", "901"}})
        linked = _fake_linker(monkeypatch, {"a": ["900"], "b": ["901"]})
        cases = [_case("a", ["10"]), _case("b", ["11"])]
        out, record = labels_for(cases, "section_aware")
        assert [c.relevant_chunk_ids for c in out] == [["900"], ["901"]]
        assert record["relinked"] is True
        assert record["stale_labels"] == 2
        assert record["cases_labelled_after_relink"] == 2
        assert linked == ["section_aware"]

    def test_the_callers_golden_set_is_never_mutated(self, monkeypatch):
        from edgar_intel.evals.evidence import labels_for

        _fake_index(monkeypatch, {})
        _fake_linker(monkeypatch, {"a": ["900"]})
        cases = [_case("a", ["10"])]
        labels_for(cases, "fixed")
        assert cases[0].relevant_chunk_ids == ["10"]

    def test_another_strategy_gets_its_own_labels(self, monkeypatch):
        """The `eval compare` case: section_aware ids mean nothing to `fixed`."""
        from edgar_intel.evals.evidence import labels_for

        _fake_index(monkeypatch, {"section_aware": {"10"}, "fixed": {"500"}})
        linked = _fake_linker(monkeypatch, {"a": ["500"]})
        out, record = labels_for([_case("a", ["10"])], "fixed")
        assert out[0].relevant_chunk_ids == ["500"]
        assert record["relinked"] and linked == ["fixed"]

    def test_a_shrinking_labelled_population_is_reported(self, monkeypatch):
        """hit@k over fewer cases is a different number, and has to say so."""
        from edgar_intel.evals.evidence import labels_for

        _fake_index(monkeypatch, {})
        _fake_linker(monkeypatch, {"a": ["900"]})
        _, record = labels_for([_case("a", ["10"]), _case("b", ["11"])], "semantic")
        assert record["cases_labelled"] == 2
        assert record["cases_labelled_after_relink"] == 1


class TestRunSuiteGradesAgainstItsOwnIndex:
    @staticmethod
    def _source() -> str:
        from edgar_intel.evals import runner

        return inspect.getsource(runner.run_suite)

    def test_labels_are_checked_before_any_case_runs(self):
        tree = ast.parse(self._source())
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "labels_for"
        ]
        loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)]
        assert len(calls) == 1, "run_suite must call labels_for exactly once"
        assert calls[0].lineno < min(loop.lineno for loop in loops)

    def test_the_relabelled_cases_are_the_ones_that_run(self):
        """Calling the check and then grading the original list would fix nothing."""
        tree = ast.parse(self._source())
        assigns = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "labels_for"
        ]
        assert assigns
        target = assigns[0].targets[0]
        assert isinstance(target, ast.Tuple) and target.elts[0].id == "cases"

    def test_the_config_records_the_index_and_the_labels(self):
        source = self._source()
        assert '"index": index' in source
        assert '"evidence_labels": evidence_labels' in source

    def test_compare_goes_through_run_suite(self):
        from edgar_intel.evals import runner

        tree = ast.parse(inspect.getsource(runner.compare_strategies))
        called = {
            n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "run_suite" in called


class TestRunSuiteRefusesAnIndexThatIsNotThere:
    """A typo in a strategy name, or an interrupted build, is not a score."""

    @staticmethod
    def _patch(monkeypatch, chunks: int, embedded: int):
        import edgar_intel.evals.runner as runner

        class Reached(Exception):
            pass

        monkeypatch.setattr(
            runner, "index_shape",
            lambda st: {"strategy": st, "chunks": chunks, "embedded": embedded},
        )
        monkeypatch.setattr(runner, "labels_for", lambda cases, st: (cases, {}))

        def insert(*a, **k):
            raise Reached

        monkeypatch.setattr(runner.db, "query_one", insert)
        return runner, Reached

    def test_an_empty_index_aborts_before_a_run_is_recorded(self, monkeypatch):
        runner, _ = self._patch(monkeypatch, chunks=0, embedded=0)
        with pytest.raises(runner.RunAborted, match="empty index"):
            runner.run_suite([_case("a", [])], strategy="section_aware_t169")

    def test_a_half_embedded_index_aborts_for_dense_and_hybrid(self, monkeypatch):
        runner, _ = self._patch(monkeypatch, chunks=100, embedded=40)
        with pytest.raises(runner.RunAborted, match="40 of 100"):
            runner.run_suite([_case("a", [])], mode="hybrid")

    def test_lexical_only_does_not_need_embeddings(self, monkeypatch):
        runner, reached = self._patch(monkeypatch, chunks=100, embedded=0)
        with pytest.raises(reached):
            runner.run_suite([_case("a", [])], mode="lexical")


class TestTheSweep:
    def test_it_refuses_to_overwrite_cited_evidence(self, tmp_path, monkeypatch):
        """The 0.194 ceiling gap and the reranker latency live in the old file."""
        from edgar_intel.evals import retrieval_eval

        out = tmp_path / "retrieval_sweep.json"
        out.write_text("{}")
        monkeypatch.setattr(
            "edgar_intel.evals.evidence.labels_for",
            lambda *a: pytest.fail("must refuse before touching the index"),
        )
        with pytest.raises(FileExistsError):
            retrieval_eval.sweep([], out_path=str(out))
        assert out.read_text() == "{}"

    def test_each_strategy_is_graded_on_labels_for_its_own_index(self, tmp_path, monkeypatch):
        from edgar_intel.evals import evidence, retrieval_eval

        relabelled = [_case("a", ["900"])]
        asked: list[str] = []

        def labels_for(cases, strategy):
            asked.append(strategy)
            return relabelled, {"strategy": strategy, "relinked": True}

        graded: list = []

        def run_config(cases, config, progress=None):
            graded.append(cases)
            return retrieval_eval.SweepResult(config=config, n_cases=len(cases))

        monkeypatch.setattr(evidence, "labels_for", labels_for)
        monkeypatch.setattr(evidence, "index_shape", lambda st: {"strategy": st})
        monkeypatch.setattr(retrieval_eval, "run_config", run_config)

        payload = retrieval_eval.sweep(
            [_case("a", ["10"])], out_path=str(tmp_path / "s.json"), strategy="section_aware_t169"
        )
        assert asked == ["section_aware_t169"]
        assert all(cases is relabelled for cases in graded)
        assert payload["evidence_labels"]["section_aware_t169"]["relinked"] is True


class TestTheDiagnosticHarnessesToo:
    @pytest.mark.parametrize("module, fn", [("context_probe", "probe"), ("autopsy", "autopsy")])
    def test_they_check_labels_before_reporting_gold_ids(self, module, fn):
        import importlib

        mod = importlib.import_module(f"edgar_intel.evals.{module}")
        assert "labels_for(" in inspect.getsource(getattr(mod, fn))
