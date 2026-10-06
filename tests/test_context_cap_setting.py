"""`eval run --context-max-chars` changes the cap, and the run records the cap that ran.

The answering context is capped at 12,000 characters, which admits about six of
the eight passages `top_n` selects. Measuring a larger cap needs the run to use
it and to say so. Before this option, the config recorded the 12,000 constant
whatever the run did, so an experiment that changed the cap would have been
written down as one that did not.
"""

from __future__ import annotations

import json

import pytest

from edgar_intel.retrieval.search import DEFAULT_CONTEXT_MAX_CHARS


def _run(monkeypatch, cases=(), **kwargs) -> tuple[dict, list[int | None]]:
    """The config a run records, and the cap each case was answered under."""
    import edgar_intel.evals.runner as runner

    monkeypatch.setattr(
        runner, "index_shape", lambda st: {"strategy": st, "chunks": 10, "embedded": 10}
    )
    monkeypatch.setattr(runner, "labels_for", lambda cases, st: (cases, {}))
    captured: dict = {}
    caps: list[int | None] = []

    class Stop(Exception):
        pass

    def query_one(sql, params=None):
        if "INSERT INTO eval_runs" in sql:
            captured["config"] = json.loads(params[3])
            return {"id": 1}
        return None

    def answer_question(case, *args, context_max_chars=None, **kw):
        caps.append(context_max_chars)
        raise Stop  # recorded as that case's error; the run carries on

    monkeypatch.setattr(runner.db, "query_one", query_one)
    monkeypatch.setattr(runner.db, "execute", lambda *a, **k: None)
    monkeypatch.setattr(runner, "answer_question", answer_question)
    runner.run_suite(list(cases), sha="test", **kwargs)
    return captured["config"], caps


@pytest.fixture
def one_case():
    from edgar_intel.evals.schemas import EvalCase

    return [EvalCase(case_id="num-X", kind="numeric", question="q?", expected="$1")]


class TestTheRunRecordsTheCap:
    def test_an_unflagged_run_records_the_shipped_cap(self, monkeypatch):
        config, _ = _run(monkeypatch)
        assert config["context_max_chars"] == DEFAULT_CONTEXT_MAX_CHARS == 12000

    def test_an_explicit_cap_is_recorded_as_given(self, monkeypatch):
        config, _ = _run(monkeypatch, context_max_chars=20000)
        assert config["context_max_chars"] == 20000


class TestTheCapReachesTheAnswer:
    def test_the_explicit_cap_is_what_each_case_is_answered_under(self, monkeypatch, one_case):
        _, caps = _run(monkeypatch, one_case, context_max_chars=20000)
        assert caps == [20000]

    def test_the_default_reaches_it_too(self, monkeypatch, one_case):
        _, caps = _run(monkeypatch, one_case)
        assert caps == [12000]
