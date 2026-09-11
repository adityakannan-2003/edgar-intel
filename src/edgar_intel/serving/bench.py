"""Load benchmark: latency percentiles, throughput, and cost per 1k requests.

Reports p50/p95/p99 rather than a mean, because a mean hides exactly the
behaviour that matters. A service averaging 400ms with a p99 of nine seconds is
a service that is visibly broken for one user in a hundred, and the mean will
never tell you.

Also measures at several concurrency levels, because the number people quote
from a single-threaded loop is the one number guaranteed not to describe
production.
"""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

DEFAULT_QUERIES = [
    "What were the main risk factors disclosed?",
    "How did revenue change year over year?",
    "What does the company say about supply chain concentration?",
    "Describe the foreign currency exposure and hedging approach.",
    "What were research and development expenses?",
    "What material legal proceedings are disclosed?",
]


@dataclass(slots=True)
class BenchResult:
    endpoint: str
    concurrency: int
    requests: int
    successes: int
    failures: int
    wall_seconds: float
    throughput_rps: float
    p50_ms: int
    p95_ms: int
    p99_ms: int
    max_ms: int
    mean_ms: int
    total_cost_usd: float = 0.0
    cost_per_1k_usd: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        return (
            f"{self.endpoint} @ concurrency {self.concurrency}\n"
            f"  requests   {self.requests} ({self.failures} failed)\n"
            f"  throughput {self.throughput_rps:.2f} req/s over {self.wall_seconds:.1f}s\n"
            f"  latency    p50 {self.p50_ms}ms  p95 {self.p95_ms}ms  "
            f"p99 {self.p99_ms}ms  max {self.max_ms}ms\n"
            f"  cost       ${self.total_cost_usd:.4f} total, "
            f"${self.cost_per_1k_usd:.3f} per 1k requests"
        )


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(q * len(sorted_values)))
    return sorted_values[idx]


def run_bench(
    base_url: str = "http://localhost:8000",
    endpoint: str = "/search",
    requests_n: int = 200,
    concurrency: int = 8,
    queries: list[str] | None = None,
    timeout: float = 60.0,
) -> BenchResult:
    queries = queries or DEFAULT_QUERIES
    url = f"{base_url.rstrip('/')}{endpoint}"

    def payload(i: int) -> dict[str, Any]:
        query = queries[i % len(queries)]
        if endpoint == "/ask":
            return {"question": query}
        return {"query": query, "top_n": 5}

    latencies: list[float] = []
    errors: list[str] = []
    total_cost = 0.0
    successes = 0

    started = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_one_request, client, url, payload(i)): i
                for i in range(requests_n)
            }
            for future in as_completed(futures):
                latency, cost, error = future.result()
                if error:
                    errors.append(error)
                else:
                    successes += 1
                    latencies.append(latency)
                    total_cost += cost
    wall = time.perf_counter() - started

    latencies.sort()
    return BenchResult(
        endpoint=endpoint,
        concurrency=concurrency,
        requests=requests_n,
        successes=successes,
        failures=len(errors),
        wall_seconds=round(wall, 2),
        # Throughput counts successes only: counting failed requests as
        # throughput is how a broken service benchmarks well.
        throughput_rps=round(successes / wall, 2) if wall else 0.0,
        p50_ms=int(_percentile(latencies, 0.50)),
        p95_ms=int(_percentile(latencies, 0.95)),
        p99_ms=int(_percentile(latencies, 0.99)),
        max_ms=int(latencies[-1]) if latencies else 0,
        mean_ms=int(statistics.fmean(latencies)) if latencies else 0,
        total_cost_usd=round(total_cost, 6),
        cost_per_1k_usd=round(total_cost / successes * 1000, 4) if successes else 0.0,
        errors=errors[:10],
    )


def _one_request(client: httpx.Client, url: str, body: dict[str, Any]):
    started = time.perf_counter()
    try:
        resp = client.post(url, json=body)
        latency = (time.perf_counter() - started) * 1000
        if resp.status_code >= 400:
            return 0.0, 0.0, f"{resp.status_code}: {resp.text[:120]}"
        data = resp.json()
        return latency, float(data.get("cost_usd", 0.0) or 0.0), ""
    except Exception as exc:
        return 0.0, 0.0, f"{type(exc).__name__}: {exc}"


def sweep(
    base_url: str = "http://localhost:8000",
    endpoint: str = "/search",
    concurrencies: tuple[int, ...] = (1, 4, 8, 16, 32),
    requests_per_level: int = 100,
    out_path: str = "reports/bench.json",
) -> dict[str, Any]:
    """Benchmark across concurrency levels and find where latency degrades.

    The useful output is the knee: the concurrency at which p95 starts rising
    faster than throughput. That number is your capacity per instance, and it
    is the honest answer to "how many users can this handle".
    """
    results = [
        run_bench(base_url, endpoint, requests_per_level, c) for c in concurrencies
    ]

    knee = None
    for prev, cur in zip(results, results[1:], strict=False):
        if prev.p95_ms <= 0:
            continue
        latency_growth = cur.p95_ms / prev.p95_ms
        throughput_growth = (
            cur.throughput_rps / prev.throughput_rps if prev.throughput_rps else 1.0
        )
        if latency_growth > 1.5 and throughput_growth < 1.2:
            knee = prev.concurrency
            break

    payload = {
        "levels": [r.as_dict() for r in results],
        "knee_concurrency": knee,
        "interpretation": (
            f"Throughput stops scaling around concurrency {knee}; beyond it, latency "
            "grows without added capacity. Treat that as per-instance capacity."
            if knee
            else "No clear saturation point within the tested range; test higher concurrency."
        ),
    }

    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload
