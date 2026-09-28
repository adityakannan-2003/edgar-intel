"""What the deployed instance serves has to match what the numbers describe.

Three things a public URL would have got wrong, found while verifying the
deploy on 27 Sep:

  `/search` defaulted to `rerank: true`, and the agent behind `/ask` reranked
  through `search()`'s own default -- while every published number was measured
  with `--no-rerank`. The service would have run the slower configuration
  baseline-v6 rejected, and `/config` could not say so.

  `/stats/eval` returned the latest run's raw summary, including the judge's
  narrative pass rate (0.75 in baseline-v6) and the overall score that blends it
  in -- numbers this project's own kappa floor says not to report.

  `/ready` counted embedded chunks of any strategy, so a database holding only
  some other index would report ready and then answer nothing.

These call the endpoint functions directly rather than through a test client:
the functions are the behaviour, and it keeps the file runnable wherever the
serving extra is installed.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi", reason="serving extra not installed")

from edgar_intel.config import reset_settings_cache  # noqa: E402
from edgar_intel.serving import app as app_module  # noqa: E402


def _set(monkeypatch, name: str, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    reset_settings_cache()


class _Result:
    def __init__(self, stage: dict) -> None:
        self.hits: list = []
        self.latency_ms = 3
        self.stage_latency_ms = stage


class TestSearchServesTheShippedConfiguration:
    def test_the_request_leaves_reranking_to_the_instance(self):
        assert app_module.SearchRequest(query="q").rerank is None

    def test_an_unspecified_request_passes_none_through(self, monkeypatch):
        seen: dict = {}

        def search(*args, **kwargs):
            seen.update(kwargs)
            return _Result({"retrieve_ms": 12})

        monkeypatch.setattr(app_module, "search", search)
        body = app_module.search_endpoint(app_module.SearchRequest(query="supplier risk"))
        assert seen["use_rerank"] is None
        assert body["reranked"] is False

    def test_the_response_says_when_the_cross_encoder_ran(self, monkeypatch):
        monkeypatch.setattr(
            app_module, "search", lambda *a, **k: _Result({"retrieve_ms": 12, "rerank_ms": 540})
        )
        body = app_module.search_endpoint(app_module.SearchRequest(query="q", rerank=True))
        assert body["reranked"] is True


class TestConfigReportsIt:
    def test_off_by_default(self, monkeypatch):
        _set(monkeypatch, "EDGAR_USE_RERANK", None)
        assert app_module.config()["use_rerank"] is False

    def test_on_when_the_instance_is_configured_so(self, monkeypatch):
        _set(monkeypatch, "EDGAR_USE_RERANK", "true")
        assert app_module.config()["use_rerank"] is True


class TestReadinessCountsTheServedIndex:
    def test_it_counts_only_the_default_strategy(self, monkeypatch):
        _set(monkeypatch, "EDGAR_DEFAULT_STRATEGY", None)
        seen: list = []

        def query_one(sql, params=None):
            seen.append((sql, params))
            return {"n": 5468}

        class Embedder:
            def embed(self, texts):
                return [[0.0] * 384]

        monkeypatch.setattr(app_module.db, "healthcheck", lambda: True)
        monkeypatch.setattr(app_module.db, "query_one", query_one)
        monkeypatch.setattr("edgar_intel.providers.get_embedder", lambda: Embedder())
        response = app_module.ready()
        body = json.loads(response.body)
        assert response.status_code == 200
        assert body["checks"]["strategy"] == "section_aware"
        assert body["checks"]["indexed_chunks"] == 5468
        sql, params = seen[0]
        assert "strategy = %s" in sql and params == ("section_aware",)


class TestStatsEvalWithholdsTheUncalibratedJudge:
    V6 = {
        "numeric_accuracy": 0.7404,
        "narrative_pass_rate": 0.75,
        "overall_score": 0.7414,
        "abstention_rate": 0.1635,
        "hallucination_rate": 0.0962,
        "judge_kappa": None,
        "judge_labels_matched": 12,
    }

    @staticmethod
    def _serve(monkeypatch, summary) -> dict:
        monkeypatch.setattr(
            app_module,
            "latest_run",
            lambda *a, **k: {
                "run_key": "baseline-v6-no-rerank-7362c6c7",
                "label": "baseline-v6-no-rerank",
                "git_sha": "cc0a69c",
                "summary": summary,
            },
        )
        return app_module.eval_stats()

    def test_baseline_v6_as_it_would_have_been_published(self, monkeypatch):
        body = self._serve(monkeypatch, dict(self.V6))
        cal = body["judge_calibration"]
        assert cal["calibrated"] is False
        assert set(cal["withheld"]) == {"narrative_pass_rate", "overall_score"}
        assert body["summary"]["narrative_pass_rate"] is None
        assert body["summary"]["overall_score"] is None
        assert "only 12 of this run's answers carry human labels" in cal["verdict"]

    def test_the_verifiable_numbers_are_untouched(self, monkeypatch):
        body = self._serve(monkeypatch, dict(self.V6))
        assert body["summary"]["numeric_accuracy"] == 0.7404
        assert body["summary"]["hallucination_rate"] == 0.0962
        assert body["git_sha"] == "cc0a69c"

    def test_a_kappa_below_the_floor_is_withheld_too(self, monkeypatch):
        """0.4167 is the shipped configuration's measured kappa."""
        body = self._serve(monkeypatch, {**self.V6, "judge_kappa": 0.4167})
        assert body["summary"]["narrative_pass_rate"] is None
        assert "BELOW" in body["judge_calibration"]["verdict"]

    def test_a_calibrated_judge_is_reported(self, monkeypatch):
        body = self._serve(monkeypatch, {**self.V6, "judge_kappa": 0.72})
        assert body["judge_calibration"]["calibrated"] is True
        assert body["judge_calibration"]["withheld"] == []
        assert body["summary"]["narrative_pass_rate"] == 0.75

    def test_a_summary_stored_as_text_is_handled(self, monkeypatch):
        body = self._serve(monkeypatch, json.dumps(self.V6))
        assert body["summary"]["narrative_pass_rate"] is None

    def test_no_runs_says_so(self, monkeypatch):
        monkeypatch.setattr(app_module, "latest_run", lambda *a, **k: None)
        assert app_module.eval_stats() == {"status": "no evaluation runs recorded"}
