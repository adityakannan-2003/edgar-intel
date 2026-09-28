"""Reranking is one setting, and it is off.

Until 27 Sep the default was a hard-coded True in `search()` and in `/search`'s
request model, so every caller that did not say -- `/search`, the agent's search
tool behind `/ask`, `eval run` without a flag -- reranked. Every published number
(baseline-v5, baseline-v6-no-rerank) was measured without reranking. A deployed
instance would have served a configuration the metrics do not describe, and the
slower one: baseline-v6 measured the cross-encoder at +540 ms p50 for 0.7356
numeric accuracy against 0.7404 without it.

`Settings.use_rerank` is now the one place that decides; everything that does
not pass an explicit value follows it, and the eval config records the value
that actually ran.
"""

from __future__ import annotations

import ast
import inspect
import json

import pytest

from edgar_intel.config import get_settings, reset_settings_cache


def _set(monkeypatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("EDGAR_USE_RERANK", raising=False)
    else:
        monkeypatch.setenv("EDGAR_USE_RERANK", value)
    reset_settings_cache()


class TestTheSetting:
    def test_it_is_off_by_default(self, monkeypatch):
        _set(monkeypatch, None)
        assert get_settings().use_rerank is False

    def test_the_environment_can_turn_it_on(self, monkeypatch):
        _set(monkeypatch, "true")
        assert get_settings().use_rerank is True


class TestSearchFollowsIt:
    """`search()` with no `use_rerank` does what the instance ships."""

    @staticmethod
    def _wire(monkeypatch) -> list[str]:
        # The package re-exports the function under the module's name, so the
        # module itself has to come from importlib.
        import importlib

        search_mod = importlib.import_module("edgar_intel.retrieval.search")

        def hits(source: str):
            return [
                search_mod.Hit(
                    chunk_id=str(i), body=f"passage {i}", filing_id=1, score=1.0,
                    source=source, ranks={source: i},
                )
                for i in range(1, 4)
            ]

        called: list[str] = []
        monkeypatch.setattr(search_mod, "dense_search", lambda *a, **k: hits("dense"))
        monkeypatch.setattr(search_mod, "lexical_search", lambda *a, **k: hits("lexical"))

        def rerank(query, candidates):
            called.append(query)
            return list(reversed(candidates))

        monkeypatch.setattr(search_mod, "rerank", rerank)
        return called

    def test_no_argument_and_the_setting_off_means_no_rerank(self, monkeypatch):
        from edgar_intel.retrieval.search import search

        _set(monkeypatch, None)
        called = self._wire(monkeypatch)
        result = search("q")
        assert called == []
        assert "rerank_ms" not in result.stage_latency_ms

    def test_no_argument_and_the_setting_on_means_rerank(self, monkeypatch):
        from edgar_intel.retrieval.search import search

        _set(monkeypatch, "true")
        called = self._wire(monkeypatch)
        result = search("q")
        assert called == ["q"]
        assert "rerank_ms" in result.stage_latency_ms

    @pytest.mark.parametrize("setting, explicit", [(None, True), ("true", False)])
    def test_an_explicit_value_still_wins(self, monkeypatch, setting, explicit):
        """Experiments -- the sweep, the probe, --rerank -- must not be overridden."""
        from edgar_intel.retrieval.search import search

        _set(monkeypatch, setting)
        called = self._wire(monkeypatch)
        search("q", use_rerank=explicit)
        assert bool(called) is explicit


class TestEveryCallerThatDoesNotSayFollowsIt:
    def test_the_agent_does_not_choose_for_itself(self):
        """`/ask` runs this tool. A hard-coded value here would diverge silently."""
        from edgar_intel.agent import tools

        tree = ast.parse(inspect.getsource(tools.search_filings))
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "search"
        ]
        assert len(calls) == 1
        assert "use_rerank" not in {kw.arg for kw in calls[0].keywords}

    @pytest.mark.parametrize("fn", ["answer_question", "run_suite"])
    def test_the_eval_path_defaults_to_the_setting(self, fn):
        from edgar_intel.evals import runner

        assert inspect.signature(getattr(runner, fn)).parameters["use_rerank"].default is None

    def test_search_itself_defaults_to_the_setting(self):
        from edgar_intel.retrieval.search import search

        assert inspect.signature(search).parameters["use_rerank"].default is None


class TestTheRunRecordsWhatRan:
    """A config saying `use_rerank: null` would be a knob nobody can rule out."""

    @staticmethod
    def _config_of_a_run(monkeypatch, **kwargs) -> dict:
        import edgar_intel.evals.runner as runner

        class Recorded(Exception):
            pass

        monkeypatch.setattr(
            runner, "index_shape", lambda st: {"strategy": st, "chunks": 10, "embedded": 10}
        )
        monkeypatch.setattr(runner, "labels_for", lambda cases, st: (cases, {}))
        captured: dict = {}

        def insert(sql, params=None):
            captured["config"] = json.loads(params[3])
            raise Recorded

        monkeypatch.setattr(runner.db, "query_one", insert)
        with pytest.raises(Recorded):
            runner.run_suite([], **kwargs)
        return captured["config"]

    def test_an_unflagged_run_records_the_shipped_value(self, monkeypatch):
        _set(monkeypatch, None)
        assert self._config_of_a_run(monkeypatch)["use_rerank"] is False

    def test_it_follows_the_setting(self, monkeypatch):
        _set(monkeypatch, "true")
        assert self._config_of_a_run(monkeypatch)["use_rerank"] is True

    def test_an_explicit_value_is_recorded_as_given(self, monkeypatch):
        _set(monkeypatch, None)
        assert self._config_of_a_run(monkeypatch, use_rerank=True)["use_rerank"] is True


class TestTheCommandLine:
    """`eval run --no-rerank` produced every published number; it must keep working."""

    @staticmethod
    def _eval_run() -> ast.FunctionDef:
        import pathlib

        tree = ast.parse(pathlib.Path("src/edgar_intel/cli.py").read_text())
        return next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "eval_run"
        )

    def test_both_flags_exist_and_neither_means_the_setting(self):
        fn = self._eval_run()
        source = ast.unparse(fn)
        assert "'--rerank'" in source and "'--no-rerank'" in source
        assert "use_rerank = True if rerank else False if no_rerank else None" in source

    def test_the_resolved_value_reaches_the_run(self):
        calls = [
            n for n in ast.walk(self._eval_run())
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "run_suite"
        ]
        kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
        assert kw["use_rerank"] == "use_rerank"


class TestTheSweepCallsTheRightRowShipped:
    """The sweep's "shipped" row reranked unconditionally, so after reranking was
    dropped its diagnosis kept calling the unshipped configuration "shipped"."""

    def test_shipped_follows_the_setting_and_the_other_row_flips_it(self, monkeypatch):
        from edgar_intel.evals.retrieval_eval import default_sweep

        _set(monkeypatch, None)
        rows = {c.name: c for c in default_sweep()}
        assert rows["shipped"].use_rerank is False
        assert rows["rerank"].use_rerank is True
        assert "no-rerank" not in rows

        _set(monkeypatch, "true")
        rows = {c.name: c for c in default_sweep()}
        assert rows["shipped"].use_rerank is True
        assert rows["no-rerank"].use_rerank is False

    @staticmethod
    def _results(shipped_rerank: bool, shipped_v: float, other_v: float):
        from edgar_intel.evals.retrieval_eval import SweepConfig, SweepResult

        other = "no-rerank" if shipped_rerank else "rerank"
        return [
            SweepResult(SweepConfig("shipped", "hybrid", shipped_rerank, 50, 8), 232,
                        {"hit@5": shipped_v}),
            SweepResult(SweepConfig(other, "hybrid", not shipped_rerank, 50, 8), 232,
                        {"hit@5": other_v}),
            SweepResult(SweepConfig("ceiling", "hybrid", False, 50, 50), 232, {"hit@20": 0.853}),
        ]

    def test_the_27_sep_numbers_read_correctly_with_reranking_off(self):
        """No-rerank ships at 0.6595; the reranked row is 0.6379 (reports print 3 dp)."""
        from edgar_intel.evals.retrieval_eval import diagnose

        text = diagnose(self._results(False, 0.6595, 0.6379))
        assert "shipped hit@5=0.659" in text
        assert "Reranking is HURTING: hit@5 is 0.659 without it versus 0.638 with it" in text
        assert "off in the shipped configuration" in text

    def test_the_old_reading_is_unchanged_when_reranking_ships(self):
        from edgar_intel.evals.retrieval_eval import diagnose

        text = diagnose(self._results(True, 0.6379, 0.6595))
        assert "Reranking is HURTING: hit@5 is 0.659 without it versus 0.638 with it" in text
        assert "consider dropping it" in text
