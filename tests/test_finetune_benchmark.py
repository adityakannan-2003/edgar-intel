"""The benchmark harness, end to end with a scripted model: no network, no spend."""

from __future__ import annotations

import json

import pytest

from edgar_intel.finetune.benchmark import (
    Arm,
    aggregate,
    compare,
    paired_bootstrap,
    pick_shots,
    predict,
    score,
    select,
    system_prompt,
)
from edgar_intel.finetune.dataset import INSTRUCTION, Example, write_jsonl
from edgar_intel.providers.base import Completion

REV = {"concept": "total revenue", "fiscal_year": 2024, "value": 391_035_000_000}
NI = {"concept": "net income", "fiscal_year": 2024, "value": 93_736_000_000}


def example(i: int, ticker: str, facts: list[dict], passage: str = "") -> Example:
    return Example(
        id=f"{ticker}:{i}", ticker=ticker, company=ticker, accession=ticker,
        filing_fy=2024, period_end="2024-09-28",
        passage=passage or f"Total net sales | 391,035\nNet income | 93,736 ({i})",
        facts=facts, sector="s",
        verifiable=[["net income", 2024], ["total revenue", 2024]],
    )


@pytest.fixture
def files(tmp_path):
    held = [example(i, "AAA", [REV, NI]) for i in range(4)] + [
        example(9, "AAA", [], passage="Risk factors, in prose.")
    ]
    train = [example(i, f"T{i}", [REV, NI]) for i in range(5)] + [
        example(9, "T9", [], passage="Debt | 1,000 | 2,000 | 3,000")
    ]
    held_path, train_path = tmp_path / "held.jsonl", tmp_path / "train.jsonl"
    write_jsonl(held, str(held_path))
    write_jsonl(train, str(train_path))
    return str(held_path), str(train_path), tmp_path


class ScriptedLLM:
    """Answers from a list; an Exception in the list is raised instead."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def complete(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        raw = {"usage": {"prompt_tokens": 1000, "completion_tokens": 50,
                         "prompt_tokens_details": {"cached_tokens": 800}}}
        return Completion(text=answer, raw=raw, model="m")


ARM = Arm("test arm", "gpt-6-luna", few_shot=True)
RIGHT = json.dumps([REV, NI])


class TestShots:
    def test_shots_come_from_distinct_training_companies_plus_a_table_negative(self, files):
        _, train, _ = files
        shots = pick_shots(train, max_chars=5000)
        tickers = [s["ticker"] for s in shots]
        assert len(set(tickers)) == len(tickers)
        assert shots[-1]["facts"] == []

    def test_zero_shot_system_prompt_is_the_instruction(self):
        assert system_prompt([]) == INSTRUCTION


class TestPredict:
    def test_every_case_is_written_and_the_input_is_the_held_out_user_message(self, files):
        held, train, tmp = files
        llm = ScriptedLLM([RIGHT] * 5)
        out = str(tmp / "p.jsonl")
        summary = predict(ARM, held, out, train_path=train, llm=llm)
        assert summary["written"] == 5
        assert llm.calls[0]["prompt"].startswith("Company: AAA (AAA)")
        assert "Worked examples" in llm.calls[0]["system"]
        assert len(open(out).read().splitlines()) == 5

    def test_resume_skips_written_cases(self, files):
        held, train, tmp = files
        out = str(tmp / "p.jsonl")
        predict(ARM, held, out, train_path=train, n=2, llm=ScriptedLLM([RIGHT] * 2))
        llm = ScriptedLLM([RIGHT] * 3)
        summary = predict(ARM, held, out, train_path=train, llm=llm)
        assert summary["skipped"] == 2 and len(llm.calls) == 3

    def test_a_file_belongs_to_one_configuration(self, files):
        held, train, tmp = files
        out = str(tmp / "p.jsonl")
        predict(ARM, held, out, train_path=train, n=1, llm=ScriptedLLM([RIGHT]))
        other = Arm("other", "gpt-6.1-sol", few_shot=True)
        with pytest.raises(ValueError, match="another configuration"):
            predict(other, held, out, train_path=train, llm=ScriptedLLM([RIGHT]))

    def test_consecutive_failures_abort_and_are_recorded(self, files):
        held, train, tmp = files
        out = str(tmp / "p.jsonl")
        llm = ScriptedLLM([RuntimeError("HTTP 401")] * 5)
        with pytest.raises(RuntimeError, match="consecutive failures"):
            predict(ARM, held, out, train_path=train, llm=llm, max_consecutive_errors=3)
        lines = [json.loads(x) for x in open(out)]
        assert len(lines) == 3 and all("error" in x for x in lines)

    def test_select_is_a_fixed_prefix_of_one_shuffle(self):
        rows = [{"id": str(i)} for i in range(100)]
        small = {r["id"] for r in select(rows, 10)}
        large = {r["id"] for r in select(rows, 30)}
        assert small < large


class TestScore:
    def test_metrics_from_a_mixed_run(self, files):
        held, train, tmp = files
        out = str(tmp / "p.jsonl")
        answers = [
            RIGHT,                                    # perfect
            json.dumps([REV]),                        # misses net income
            "no json here",                           # parse failure
            RuntimeError("timeout"),                  # error, not scored
            json.dumps([NI]),                         # a fact on the negative
        ]
        predict(ARM, held, out, train_path=train, llm=ScriptedLLM(answers))
        result = score(out, held)
        m = result["metrics"]
        assert (m["cases"], m["errors"]) == (5, 1)
        assert m["micro"]["tp"] == 3 and m["micro"]["n_expected"] == 6
        assert m["parse_failures"] == 1
        assert m["negatives_with_any_fact"] == 1
        assert m["per_concept"]["total revenue"]["tp"] == 2
        # 5 calls x (200 uncached x $0.10 + 800 cached x $0.01 + 50 x $0.50) / 1M;
        # the errored call reports no tokens.
        assert m["cost"]["total_usd"] == pytest.approx(4 * (200 * 0.10 + 800 * 0.01 + 50 * 0.50) / 1e6)

    def test_a_predictions_file_cannot_be_scored_against_another_set(self, files, tmp_path):
        held, train, tmp = files
        out = str(tmp / "p.jsonl")
        predict(ARM, held, out, train_path=train, n=1, llm=ScriptedLLM([RIGHT]))
        other = tmp_path / "other.jsonl"
        write_jsonl([example(1, "ZZZ", [REV])], str(other))
        with pytest.raises(ValueError, match="another held-out set"):
            score(out, str(other))

    def test_self_hosted_cost_is_gpu_price_times_latency(self):
        cases = [{"id": "a", "ticker": "A", "negative": False, "latency_ms": 3600, "tp": 1,
                  "n_pred": 1, "n_expected": 1, "f1": 1.0, "exact_set_match": 1.0, "parse_ok": True,
                  "invalid": 0, "ignored": 0, "diagnosis": {}, "by_concept": {},
                  "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}]
        cost = aggregate(cases, "extract", gpu_usd_per_hour=1.0)["cost"]
        # 3.6 s per request on a $1/hr GPU: 1,000 requests take one hour.
        assert cost["per_1k_requests_usd"] == pytest.approx(1.0)


class TestCompare:
    def test_identical_arms_differ_by_zero(self, files):
        held, train, tmp = files
        a, b = str(tmp / "a.jsonl"), str(tmp / "b.jsonl")
        predict(ARM, held, a, train_path=train, llm=ScriptedLLM([RIGHT] * 5))
        predict(Arm("copy", "gpt-6-luna"), held, b, train_path=train, llm=ScriptedLLM([RIGHT] * 5))
        payload = compare([a, b], held, str(tmp / "report.json"))
        row = payload["vs_first"][0]
        assert row["f1_diff"] == 0 and row["ci95"] == [0, 0] and row["cases_flipped"] == 0
        assert (tmp / "report.cases.jsonl").exists()

    def test_bootstrap_is_deterministic(self):
        a = [{"id": str(i), "tp": i % 2, "n_pred": 1, "n_expected": 1, "exact_set_match": float(i % 2)}
             for i in range(20)]
        b = [{**c, "tp": 1, "exact_set_match": 1.0} for c in a]
        assert paired_bootstrap(a, b) == paired_bootstrap(a, b)
        assert paired_bootstrap(a, b)["f1_diff"] == 0.5


class TestThroughput:
    def test_requests_per_second_at_concurrency(self, files):
        from edgar_intel.finetune.benchmark import measure_throughput

        held, train, _ = files
        result = measure_throughput(
            Arm("lora", "extract", few_shot=False, self_hosted=True), held, "http://x/v1",
            n=5, concurrency=2, llm=ScriptedLLM([RIGHT] * 5),
        )
        assert result["requests"] == 5 and result["requests_per_s"] > 0

    def test_at_load_cost_is_gpu_price_over_throughput(self):
        cases = [{"id": "a", "ticker": "A", "negative": False, "latency_ms": 3600, "tp": 1,
                  "n_pred": 1, "n_expected": 1, "f1": 1.0, "exact_set_match": 1.0, "parse_ok": True,
                  "invalid": 0, "ignored": 0, "diagnosis": {}, "by_concept": {},
                  "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}]
        cost = aggregate(cases, "extract", 0.72, {"requests_per_s": 2.0, "concurrency": 16})["cost"]
        # $0.72/hr at 7,200 requests/hr is $0.10 per 1,000.
        assert cost["at_load"]["per_1k_requests_usd"] == pytest.approx(0.10)


class TestMicro:
    def test_nothing_to_score_is_none_not_zero(self):
        from edgar_intel.finetune.benchmark import _micro

        assert _micro([{"tp": 0, "n_pred": 0, "n_expected": 0}])["f1"] is None
        assert _micro([{"tp": 0, "n_pred": 1, "n_expected": 0}])["precision"] == 0.0


class TestRegrade:
    def test_corrected_labels_rescore_stored_answers_without_a_call(self, files, tmp_path):
        held, train, tmp = files
        out = str(tmp / "p.jsonl")
        predict(ARM, held, out, train_path=train, llm=ScriptedLLM([RIGHT] * 5))
        rows = [json.loads(x) for x in open(held)]
        rows[0]["facts"] = rows[0]["facts"][:1]          # a label corrected
        fixed = tmp_path / "fixed.jsonl"
        fixed.write_text("".join(json.dumps(r) + "\n" for r in rows))
        result = score(out, str(fixed))
        assert result["regraded"] and result["metrics"]["cases"] == 5

    def test_changed_inputs_are_refused(self, files, tmp_path):
        held, train, tmp = files
        out = str(tmp / "p.jsonl")
        predict(ARM, held, out, train_path=train, n=1, llm=ScriptedLLM([RIGHT]))
        rows = [json.loads(x) for x in open(held)]
        rows[0]["messages"][1]["content"] += " changed"
        other = tmp_path / "other.jsonl"
        other.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with pytest.raises(ValueError, match="another held-out set"):
            score(out, str(other))
