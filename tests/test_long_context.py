"""RAG against long context: request shape, cost, run bookkeeping, and the
report arithmetic. No database, no network: the client and the two context
builders are replaced."""

from __future__ import annotations

import json

import pytest

from edgar_intel.evals import runner
from edgar_intel.evals.schemas import EvalCase
from edgar_intel.experiments import long_context as lc
from edgar_intel.retrieval.search import Hit, RetrievalResult


def case(case_id: str, ticker: str = "AAPL", year: int = 2025, kind: str = "numeric",
         value: float | None = 416_161_000_000.0, difficulty: str = "single_hop") -> EvalCase:
    return EvalCase(
        case_id=case_id, kind=kind, question=f"What was {ticker}'s revenue in {year}?",
        expected="$416.16 billion", expected_value=value, unit="USD",
        ticker=ticker, fiscal_year=year, difficulty=difficulty,
    )


def responses_payload(answer: str, *, inp: int = 2000, cached: int = 0, write: int = 0) -> dict:
    return {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [
                {"type": "output_text", "text": json.dumps({"answer": answer, "citations": [1], "confidence": 0.9})}
            ]},
        ],
        "usage": {
            "input_tokens": inp,
            "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": write},
            "output_tokens": 120,
            "output_tokens_details": {"reasoning_tokens": 80},
        },
    }


class FakeClient:
    """Stands in for OpenAICompatLLM: reads the cache on a filing's second question."""

    def __init__(self, answer: str = "$416.16 billion") -> None:
        self.bodies: list[dict] = []
        self.answer = answer
        self.seen: set[str] = set()

    def _post(self, path: str, body: dict) -> dict:
        assert path == "/responses"
        self.bodies.append(body)
        key = body.get("prompt_cache_key")
        if key and key in self.seen:
            return responses_payload(self.answer, cached=1900)
        if key:
            self.seen.add(key)
            return responses_payload(self.answer, write=1900)
        return responses_payload(self.answer)


def test_split_prompt_rejoins_to_the_eval_run_prompt():
    head, tail = lc.split_prompt("[1] AAPL FY2025 Item 7\nNet sales 416,161", "What was revenue?")
    assert head + tail == runner.ANSWER_TEMPLATE.format(
        context="[1] AAPL FY2025 Item 7\nNet sales 416,161", question="What was revenue?"
    )
    assert head.endswith("Net sales 416,161\n\n")
    assert tail.startswith("QUESTION:\nWhat was revenue?")


def test_only_the_long_context_arm_marks_a_cache_breakpoint():
    common = dict(effort="medium", max_output_tokens=100, cache_key="acc-1")
    cached = lc.request_body("gpt-6-luna", "ctx", "q?", cache=True, **common)
    plain = lc.request_body("gpt-6-luna", "ctx", "q?", cache=False, **common)

    assert cached["input"][1]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert cached["prompt_cache_key"] == "acc-1"
    assert "prompt_cache_breakpoint" not in plain["input"][1]["content"][0]
    assert "prompt_cache_key" not in plain
    # Explicit mode in both, so the RAG arm writes nothing; the text is identical.
    assert cached["prompt_cache_options"] == plain["prompt_cache_options"] == {"mode": "explicit"}
    texts = lambda b: [b["input"][0]["content"]] + [m["content"][0]["text"] for m in b["input"][1:]]  # noqa: E731
    assert texts(cached) == texts(plain)
    assert cached["input"][0]["content"] == runner.ANSWER_SYSTEM


def test_parse_response_skips_reasoning_items_and_reads_usage():
    text, usage = lc.parse_response(responses_payload("$1", inp=5000, cached=4096, write=0))
    assert json.loads(text)["answer"] == "$1"
    assert usage == {"input_tokens": 5000, "cached_tokens": 4096, "cache_write_tokens": 0,
                     "output_tokens": 120, "reasoning_tokens": 80}


def test_cost_prices_reads_writes_and_uncached_input_apart():
    usage = {"input_tokens": 100_000, "cached_tokens": 60_000, "cache_write_tokens": 30_000,
             "output_tokens": 1_000, "reasoning_tokens": 800}
    # luna: 10k uncached x 0.10 + 60k read x 0.01 + 30k write x 0.125 + 1k out x 0.50
    assert lc.cost_usd("gpt-6-luna", usage) == pytest.approx((1000 + 600 + 3750 + 500) / 1e6)
    with pytest.raises(ValueError):
        lc.cost_usd("gpt-6-luna", {**usage, "input_tokens": 300_000})


def test_order_by_filing_groups_filings_and_keeps_set_order():
    cases = [case("a", "MSFT", 2025), case("b", "AAPL", 2024), case("c", "MSFT", 2025), case("d", "AAPL", 2024)]
    assert [c.case_id for c in lc.order_by_filing(cases)] == ["b", "d", "a", "c"]


def test_long_context_run_loads_each_filing_once_resumes_and_binds_config(tmp_path, monkeypatch):
    loads: list[tuple[str, int]] = []

    def fake_filing(ticker, year):
        loads.append((ticker, year))
        return f"acc-{ticker}-{year}", f"[1] {ticker} FY{year} Item 8\nTotal net sales 416,161"

    monkeypatch.setattr(lc, "filing_context", fake_filing)
    cases = [case("n1", "AAPL", 2025), case("x", "MSFT", 2025, value=1.0),
             case("n2", "AAPL", 2025), case("nar", "AAPL", 2025, kind="narrative", value=None)]
    out = str(tmp_path / "lc.jsonl")
    client = FakeClient()

    records = lc.run_arm(cases, "long_context", "gpt-6-luna", out, llm=client)

    assert loads == [("AAPL", 2025), ("MSFT", 2025)]          # grouped, one load each
    assert [r["case_id"] for r in records] == ["n1", "n2", "nar", "x"]
    by_id = {r["case_id"]: r for r in records}
    assert by_id["n1"]["passed"] and by_id["n2"]["passed"] and not by_id["x"]["passed"]
    assert by_id["nar"]["passed"] is None
    assert by_id["n1"]["first_of_filing"] and not by_id["n2"]["first_of_filing"]
    assert by_id["n2"]["usage"]["cached_tokens"] == 1900
    assert all(b["input"][1]["content"][0].get("prompt_cache_breakpoint") for b in client.bodies)

    summary = lc.summarise(lc.load_records(out))
    assert summary["numeric_passed"] == 2 and summary["numeric"] == 3
    assert summary["cache"]["requests_with_cache_read"] == 2       # n2 and nar
    assert summary["cache"]["requests_with_cache_write"] == 2      # one per filing
    assert summary["cache"]["misses_after_first"] == []

    # Resuming makes no call for answered cases; another configuration is refused.
    assert lc.run_arm(cases, "long_context", "gpt-6-luna", out, llm=client) == []
    assert len(client.bodies) == 4
    with pytest.raises(ValueError, match="different configuration"):
        lc.run_arm(cases, "long_context", "gpt-6-luna", out, llm=client, effort="low")


def test_failed_calls_are_recorded_retried_on_resume_and_abort_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "filing_context", lambda t, y: ("acc", "ctx"))

    class Down:
        def _post(self, path, body):
            raise RuntimeError("HTTP 500: upstream")

    cases = [case(f"c{i}") for i in range(7)]
    out = str(tmp_path / "lc.jsonl")
    with pytest.raises(RuntimeError, match="consecutive failed calls"):
        lc.run_arm(cases, "long_context", "gpt-6-luna", out, llm=Down())
    assert len(lc.load_records(out)) == lc.MAX_CONSECUTIVE_ERRORS

    lc.run_arm(cases, "long_context", "gpt-6-luna", out, llm=FakeClient())
    records = lc.load_records(out)
    assert len(records) == 7 and not any(r.get("error") for r in records)


def test_rag_context_is_the_context_eval_run_builds(monkeypatch):
    hits = [
        Hit(chunk_id=str(i), body=f"Total net sales {i} 416,161 " * 40, filing_id=1, score=1.0,
            source="fused", cik="0000320193", ticker="AAPL", fiscal_year=2025, item="7")
        for i in range(12)
    ]
    fixed = RetrievalResult(hits=hits, latency_ms=5, stage_latency_ms={"retrieve_ms": 5})
    calls: list[dict] = []

    def fake_search(query, **kwargs):
        calls.append(kwargs)
        return fixed

    monkeypatch.setattr(runner, "search", fake_search)
    monkeypatch.setattr(lc, "search", fake_search)
    c = case("n1")
    trace: dict = {}
    runner.answer_question(c, "section_aware", "hybrid", False, trace=trace)
    context, _ = lc.rag_context(c, "section_aware", False)

    assert "".join(lc.split_prompt(context, c.question)) == trace["prompt"]
    # Same retrieval call: the knobs answer_question leaves at None are left unset here.
    assert {k: v for k, v in calls[0].items() if v is not None} == {k: v for k, v in calls[1].items() if v is not None}


def test_summary_intervals_and_paired_difference():
    def rec(cid, passed, abstained=False):
        return {"case_id": cid, "kind": "numeric", "difficulty": "single_hop", "passed": passed,
                "abstained": abstained, "status": "completed", "first_of_filing": False,
                "latency_ms": 100, "model_ms": 90, "cost_usd": 0.001,
                "usage": {"input_tokens": 10, "cached_tokens": 0, "cache_write_tokens": 0,
                          "output_tokens": 1, "reasoning_tokens": 0}}

    a = [rec("1", True), rec("2", False, abstained=True), rec("3", False), rec("4", True)]
    b = [rec("1", True), rec("2", True), rec("3", True), rec("4", False)]
    s = lc.summarise(a)
    assert s["numeric_accuracy"] == 0.5
    assert s["abstention_rate"] == 0.25 and s["wrong_answer_rate"] == 0.25
    lo, hi = s["numeric_accuracy_ci95"]
    assert lo < 0.5 < hi and 0.0 <= lo and hi <= 1.0
    assert lc.wilson(0, 10)[0] == 0.0 and lc.wilson(10, 10)[1] == 1.0

    p = lc.paired(a, b)
    assert p["accuracy_diff"] == 0.25
    assert p["passed_only_in_b"] == ["2", "3"] and p["passed_only_in_a"] == ["4"]
    assert p["ci95"][0] <= 0.25 <= p["ci95"][1]
