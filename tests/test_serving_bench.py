"""The load bench with a stubbed request: no server, no spend."""

from __future__ import annotations

import pytest

from edgar_intel.serving import bench


def test_replays_bodies_in_order_and_prices_rent_by_wall_time(monkeypatch):
    sent = []

    def one(client, url, body):
        sent.append(body["id"])
        return 5.0, 0.0, ""

    monkeypatch.setattr(bench, "_one_request", one)
    r = bench.run_bench(
        "http://pod", "/v1/chat/completions", requests_n=5, concurrency=1,
        bodies=[{"id": 0}, {"id": 1}], gpu_usd_per_hour=3600.0,  # $1 a second
    )
    assert sent == [0, 1, 0, 1, 0]
    assert r.successes == 5
    assert r.total_cost_usd == pytest.approx(r.wall_seconds, abs=0.01)
    assert r.cost_per_1k_usd == pytest.approx(r.total_cost_usd / 5 * 1000, rel=1e-3)
