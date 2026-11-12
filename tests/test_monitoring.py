"""`/stats/monitoring` and `/dashboard`: what the monitoring page is drawn from.

No database: `db.query` / `db.query_one` are replaced by a fake that answers by
which table the SQL reads, so these run in the CI test job, which has none.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="serving extra not installed")
from fastapi.testclient import TestClient  # noqa: E402

from edgar_intel.serving import app as app_module  # noqa: E402
from edgar_intel.serving import monitoring  # noqa: E402

T0 = datetime(2026, 10, 22, 18, 0, tzinfo=UTC)

TRACES = [  # newest first, as `ORDER BY id DESC` returns them
    {"created_at": T0, "outcome": "answered", "total_latency_ms": 9000, "total_cost_usd": 0.0006},
    {"created_at": T0, "outcome": "answered", "total_latency_ms": 7000, "total_cost_usd": 0.0004},
    {"created_at": T0, "outcome": "escalated", "total_latency_ms": 8000, "total_cost_usd": 0.0005},
]
EVAL_RUN = {
    "run_key": "exp-prompt-conventions-c0f6e105", "label": "exp-prompt-conventions",
    "git_sha": "bee5dc0", "finished_at": T0, "n_cases": 232, "pipeline": "rag",
    "numeric_accuracy": 0.8606, "abstention_rate": 0.0865, "hallucination_rate": 0.0529,
    "hit_at_5": 0.7619, "mrr": 0.5552, "cost_per_1k_usd": 0.7804,
}
FILINGS = {"filings": 24, "newest_ingested_at": T0, "newest_filing_date": date(2026, 8, 1)}


class FakeDB:
    def __init__(self, traces=TRACES, refresh_table=False, refresh_rows=()):
        self.traces, self.refresh_table, self.refresh_rows = traces, refresh_table, refresh_rows
        self.sql: list[str] = []

    def query(self, sql, params=None):
        self.sql.append(sql)
        if "FROM agent_traces" in sql:
            return list(self.traces)
        if "FROM eval_runs" in sql:
            assert params == (monitoring._MIN_EVAL_CASES,)
            return [EVAL_RUN]
        if "FROM refresh_runs" in sql:
            rows = [r for r in self.refresh_rows if "status = 'ok'" not in sql or r["status"] == "ok"]
            return rows[:1]
        raise AssertionError(f"unexpected query: {sql}")

    def query_one(self, sql, params=None):
        if "to_regclass" in sql:
            return {"t": "refresh_runs" if self.refresh_table else None}
        if "FROM filings" in sql:
            return FILINGS
        rows = self.query(sql, params)
        return rows[0] if rows else None


@pytest.fixture
def client(monkeypatch):
    for window in monitoring._latency.values():
        window.clear()
    app_module.reset_rate_limits()
    with TestClient(app_module.app) as c:
        yield c


def use_db(monkeypatch, fake: FakeDB) -> FakeDB:
    monkeypatch.setattr(monitoring.db, "query", fake.query)
    monkeypatch.setattr(monitoring.db, "query_one", fake.query_one)
    return fake


class TestLatencyWindow:
    def test_posts_to_monitored_endpoints_are_timed_and_errors_kept_apart(
        self, client, monkeypatch
    ):
        use_db(monkeypatch, FakeDB())
        result = SimpleNamespace(hits=[], stage_latency_ms={}, latency_ms=3)
        monkeypatch.setattr(app_module, "search", lambda *a, **k: result)

        assert client.post("/search", json={"query": "revenue"}).status_code == 200
        assert client.post("/search", json={"query": "net income"}).status_code == 200
        assert client.post("/ask", json={"question": "   "}).status_code == 422
        # A bare POST is not a valid MCP request; it proves the path is the one measured.
        assert client.post("/mcp", json={}).status_code >= 400
        client.get("/health")

        lat = client.get("/stats/monitoring").json()["latency"]["endpoints"]
        assert lat["/search"]["requests"] == 2 and lat["/search"]["ok"] == 2
        assert lat["/search"]["p50_ms"] is not None and lat["/search"]["p95_ms"] is not None
        # A rejected request is not a latency sample: fast 4xx would drag p50 down.
        assert lat["/ask"] == {
            "requests": 1, "ok": 0, "client_errors": 1, "server_errors": 0,
            "p50_ms": None, "p95_ms": None,
        }
        assert lat["/mcp"]["requests"] == 1 and lat["/mcp"]["ok"] == 0
        assert "/health" not in lat

    def test_an_empty_window_is_no_data_not_zero(self, client, monkeypatch):
        use_db(monkeypatch, FakeDB())
        lat = client.get("/stats/monitoring").json()["latency"]["endpoints"]
        assert all(e["requests"] == 0 and e["p50_ms"] is None for e in lat.values())

    def test_the_window_is_bounded(self):
        assert all(w.maxlen == monitoring._WINDOW for w in monitoring._latency.values())


class TestCostPerRequest:
    def test_cost_and_latency_percentiles_come_from_agent_traces(self, client, monkeypatch):
        use_db(monkeypatch, FakeDB())
        cost = client.get("/stats/monitoring").json()["cost"]
        assert cost["runs"] == 3
        assert cost["mean_usd"] == pytest.approx(0.0005)
        assert cost["per_1k_usd"] == pytest.approx(0.5)
        assert cost["total_usd"] == pytest.approx(0.0015)
        assert cost["p50_usd"] == pytest.approx(0.0005)
        assert cost["p95_usd"] == pytest.approx(0.0006)
        assert cost["p50_latency_ms"] == 8000 and cost["p95_latency_ms"] == 9000

    def test_recent_runs_carry_no_question_or_trace_id(self, client, monkeypatch):
        use_db(monkeypatch, FakeDB())
        recent = client.get("/stats/monitoring").json()["cost"]["recent"]
        assert len(recent) == 3
        assert set(recent[0]) == {"created_at", "outcome", "latency_ms", "cost_usd"}

    def test_no_runs_is_no_data(self, client, monkeypatch):
        use_db(monkeypatch, FakeDB(traces=[]))
        cost = client.get("/stats/monitoring").json()["cost"]
        assert cost["runs"] == 0 and cost["mean_usd"] is None and cost["p95_usd"] is None


class TestEvalHistory:
    def test_runs_are_served_and_judge_fields_never_selected(self, client, monkeypatch):
        fake = use_db(monkeypatch, FakeDB())
        body = client.get("/stats/monitoring").json()
        assert body["eval"]["runs"][0]["numeric_accuracy"] == 0.8606
        eval_sql = next(s for s in fake.sql if "FROM eval_runs" in s)
        # /stats/eval withholds these until the judge is calibrated; this never selects them.
        assert "narrative_pass_rate" not in eval_sql and "overall_score" not in eval_sql


class TestFreshness:
    def test_job_is_pending_until_the_refresh_table_exists(self, client, monkeypatch):
        use_db(monkeypatch, FakeDB(refresh_table=False))
        fresh = client.get("/stats/monitoring").json()["freshness"]
        assert fresh["job"]["status"] == "pending"
        assert fresh["data"]["filings"] == 24
        assert fresh["data"]["newest_filing_date"] == "2026-08-01"

    def test_last_run_and_last_successful_run_are_reported(self, client, monkeypatch):
        failed = {"started_at": T0, "finished_at": T0, "status": "failed", "filings_added": 0}
        ok = {"started_at": datetime(2026, 10, 1, tzinfo=UTC), "finished_at": None,
              "status": "ok", "filings_added": 2}
        use_db(monkeypatch, FakeDB(refresh_table=True, refresh_rows=[failed, ok]))
        job = client.get("/stats/monitoring").json()["freshness"]["job"]
        assert job["status"] == "recorded"
        assert job["last_run"]["status"] == "failed"
        assert job["last_ok"]["filings_added"] == 2

    def test_table_with_no_rows_says_so(self, client, monkeypatch):
        use_db(monkeypatch, FakeDB(refresh_table=True))
        job = client.get("/stats/monitoring").json()["freshness"]["job"]
        assert job["status"] == "no runs yet" and job["last_ok"] is None


class TestDatabaseDown:
    def test_latency_is_still_served_and_the_error_names_only_its_type(
        self, client, monkeypatch
    ):
        def down(*a, **k):
            raise ConnectionError("connection to server at ep-secret-host.neon.tech failed")

        monkeypatch.setattr(monitoring.db, "query", down)
        r = client.get("/stats/monitoring")
        assert r.status_code == 200
        body = r.json()
        assert body["database_error"] == "ConnectionError"
        assert "neon.tech" not in r.text
        assert set(body["latency"]["endpoints"]) == {"/ask", "/search", "/mcp"}


def test_dashboard_page_draws_the_monitoring_json(client):
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert 'fetch("/stats/monitoring"' in r.text
