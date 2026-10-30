"""The model router: arm specs, pricing, each policy's routing, resume and scoring.

No database and no network: `run_agent` and the provider are stubbed, and the
scorer reads results files written here.
"""

from __future__ import annotations

import json

import pytest

from edgar_intel.agent.loop import AgentRun
from edgar_intel.evals import router
from edgar_intel.evals.schemas import EvalCase
from edgar_intel.finetune.benchmark import _OPENAI_REASONING
from edgar_intel.providers.base import Completion

CHEAP, STRONG = "gpt-6-luna", "gpt-6.1-sol"


class RecordingLLM:
    """Replies with fixed text and keeps the keyword arguments of each call."""

    name = "recording"

    def __init__(self, text: str = "{}", usage: dict | None = None) -> None:
        self.text, self.usage, self.kwargs = text, usage or {}, []

    def complete(self, prompt, **kwargs) -> Completion:
        self.kwargs.append(kwargs)
        return Completion(text=self.text, prompt_tokens=100, completion_tokens=20,
                          latency_ms=5, model=kwargs["model"], raw={"usage": self.usage})


def _run(outcome: str, answer: str = "$1 billion") -> AgentRun:
    return AgentRun(trace_id="t", question="q", outcome=outcome, answer=answer, citations=[],
                    confidence=0.9, steps=[], total_latency_ms=1, total_cost_usd=0.0)


@pytest.fixture
def stub_agent(monkeypatch):
    """`run_agent` returns the outcome scripted for the model it was given."""

    def _apply(outcomes: dict[str, str], rating_text: str = '{"difficulty": "easy"}'):
        seen: list[str] = []

        def fake_run_agent(question, persist=True, llm=None, **_):
            seen.append(llm.model)
            llm.complete("p", system="s")  # one metered call per agent run
            return _run(outcomes[llm.model])

        monkeypatch.setattr(router, "run_agent", fake_run_agent)
        monkeypatch.setattr(router, "get_llm", lambda: RecordingLLM(rating_text))
        return seen

    return _apply


# ------------------------------------------------------------------ arms
def test_parse_arm_accepts_the_three_policies():
    assert router.parse_arm("single:gpt-4o-mini") == router.Arm("single", "gpt-4o-mini")
    arm = router.parse_arm(f"cascade:{CHEAP}, {STRONG}")
    assert (arm.cheap, arm.strong, arm.spec) == (CHEAP, STRONG, f"cascade:{CHEAP},{STRONG}")
    assert router.parse_arm(f"difficulty:{CHEAP},{STRONG}").policy == "difficulty"


@pytest.mark.parametrize("spec", [
    "single", f"single:{CHEAP},{STRONG}", f"cascade:{CHEAP}", f"vote:{CHEAP},{STRONG}",
])
def test_parse_arm_rejects_malformed_specs(spec):
    with pytest.raises(ValueError, match="want single"):
        router.parse_arm(spec)


def test_parse_arm_rejects_a_model_with_no_list_price():
    with pytest.raises(ValueError, match="no list price for gpt-9"):
        router.parse_arm(f"cascade:gpt-9,{STRONG}")


# --------------------------------------------------------------- pricing
def test_call_cost_prices_cached_input_apart():
    call = {"model": CHEAP, "prompt": 1000, "cached": 400, "completion": 200}
    # luna: $0.10 fresh, $0.01 cached, $0.50 output per 1M
    assert router.call_cost(call) == pytest.approx((600 * 0.10 + 400 * 0.01 + 200 * 0.50) / 1e6)


def test_metered_llm_sends_gpt6_overrides_and_logs_usage():
    inner = RecordingLLM(usage={"prompt_tokens_details": {"cached_tokens": 64},
                                "completion_tokens_details": {"reasoning_tokens": 12}})
    plain = RecordingLLM()
    calls: list[dict] = []
    router.MeteredLLM(STRONG, calls, inner).complete("p", temperature=0.0, max_tokens=800)
    router.MeteredLLM("gpt-4o-mini", calls, plain).complete("p")

    assert inner.kwargs[0]["extra_body"] is _OPENAI_REASONING
    assert inner.kwargs[0]["model"] == STRONG
    assert "extra_body" not in plain.kwargs[0]
    assert calls[0] == {"model": STRONG, "served": STRONG, "prompt": 100, "cached": 64,
                        "completion": 20, "reasoning": 12, "latency_ms": 5}
    assert calls[1]["cached"] == 0 and calls[1]["reasoning"] == 0


# --------------------------------------------------------------- policies
def test_single_runs_one_model(stub_agent):
    seen = stub_agent({CHEAP: "escalated"})
    routed = router.answer(router.parse_arm(f"single:{CHEAP}"), "q", calls := [])
    assert seen == [CHEAP] and routed.route == [CHEAP] and routed.run.outcome == "escalated"
    assert len(calls) == 1


def test_cascade_keeps_an_answer_from_the_cheap_model(stub_agent):
    seen = stub_agent({CHEAP: "answered", STRONG: "answered"})
    routed = router.answer(router.parse_arm(f"cascade:{CHEAP},{STRONG}"), "q", [])
    assert seen == [CHEAP] and routed.outcomes == ["answered"]


@pytest.mark.parametrize("outcome", ["escalated", "exhausted", "failed"])
def test_cascade_hands_a_non_answer_to_the_strong_model(stub_agent, outcome):
    seen = stub_agent({CHEAP: outcome, STRONG: "answered"})
    routed = router.answer(router.parse_arm(f"cascade:{CHEAP},{STRONG}"), "q", calls := [])
    assert seen == [CHEAP, STRONG]
    assert routed.route == [CHEAP, STRONG] and routed.outcomes == [outcome, "answered"]
    assert routed.run.outcome == "answered"          # the strong run is the one graded
    assert [c["model"] for c in calls] == [CHEAP, STRONG]  # both runs are paid for


def test_cascade_does_not_escalate_a_screened_refusal(stub_agent):
    seen = stub_agent({CHEAP: "refused", STRONG: "answered"})
    router.answer(router.parse_arm(f"cascade:{CHEAP},{STRONG}"), "q", [])
    assert seen == [CHEAP]


@pytest.mark.parametrize("reply, rating, model", [
    ('{"difficulty": "easy"}', "easy", CHEAP),
    ('{"difficulty": "hard"}', "hard", STRONG),
    ('{"difficulty": "medium"}', "unrated", STRONG),
    ("not json", "unrated", STRONG),
    ("[1, 2]", "unrated", STRONG),
])
def test_difficulty_routes_on_the_rating(stub_agent, reply, rating, model):
    seen = stub_agent({CHEAP: "answered", STRONG: "answered"}, rating_text=reply)
    routed = router.answer(router.parse_arm(f"difficulty:{CHEAP},{STRONG}"), "q", calls := [])
    assert routed.rating == rating and seen == [model]
    assert [c["model"] for c in calls] == [CHEAP, model]  # the rating call is paid for too


# --------------------------------------------------------------- running
def _case(cid: str, kind: str = "numeric", difficulty: str = "single_hop") -> EvalCase:
    return EvalCase(case_id=cid, kind=kind, question=f"question {cid}?", expected="1",
                    difficulty=difficulty)


def test_run_router_resumes_only_the_errored_cases(monkeypatch, tmp_path):
    monkeypatch.setattr(router, "index_shape", lambda strategy: {"chunks": 1, "embedded": 1})
    attempts: list[str] = []

    def fake_row(arm, case):
        attempts.append(case.case_id)
        if case.case_id == "b" and attempts.count("b") == 1:
            return {"case_id": "b", "error": "boom", "calls": [], "cost_usd": 0.001}
        return {"case_id": case.case_id, "passed": True}

    monkeypatch.setattr(router, "_case_row", fake_row)
    cases = [_case("a"), _case("b"), _case("c")]
    spec = f"single:{CHEAP}"
    path = router.run_router(spec, cases, "r1", out_dir=str(tmp_path))
    router.run_router(spec, cases, "r1", out_dir=str(tmp_path))

    assert attempts == ["a", "b", "c", "b"]
    with pytest.raises(ValueError, match="another arm"):
        router.run_router(f"single:{STRONG}", cases, "r1", out_dir=str(tmp_path))
    assert json.load(open(path + ".meta.json"))["case_ids"] == ["a", "b", "c"]


def test_sample_cases_is_stratified_seeded_and_in_set_order():
    cases = ([_case(f"s{i}") for i in range(70)] + [_case(f"c{i}", difficulty="comparative")
             for i in range(20)] + [_case(f"n{i}", kind="narrative") for i in range(10)])
    picked = router.sample_cases(cases, 20)
    assert [c.case_id for c in picked] == [c.case_id for c in router.sample_cases(cases, 20)]
    assert [sum(c.case_id[0] == k for c in picked) for k in "scn"] == [14, 4, 2]
    order = [c.case_id for c in cases]
    assert [order.index(c.case_id) for c in picked] == sorted(order.index(c.case_id) for c in picked)


# --------------------------------------------------------------- scoring
def _row(cid: str, passed: bool, cost: float = 0.001, kind: str = "numeric", **kw) -> dict:
    return {"case_id": cid, "kind": kind, "difficulty": "single_hop", "passed": passed,
            "abstained": False, "outcome": "answered", "route": kw.get("route", [CHEAP]),
            "rating": "", "calls": [{"prompt": 10, "cached": 0, "completion": 2, "reasoning": 1}],
            "cost_usd": cost, "judge_cost_usd": 0.0, "latency_ms": kw.get("latency", 1000)}


def _write(tmp_path, label: str, arm: str, rows: list[dict]) -> str:
    path = str(tmp_path / f"{label}.jsonl")
    with open(path + ".meta.json", "w") as fh:
        json.dump({"label": label, "arm": arm, "git_sha": "abc"}, fh)
    with open(path, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    return path


def test_score_reports_accuracy_noise_cost_and_the_claim_rule(tmp_path):
    ids = [f"n{i}" for i in range(20)]
    cheap, strong = f"single:{CHEAP}", f"single:{STRONG}"
    # cheap run 1 passes 10; run 2 passes 12 (flips n10, n11 up); a resumed row
    # for n0 supersedes its error. strong passes all 20 at ten times the cost.
    c1 = [{"case_id": "n0", "kind": "numeric", "error": "boom", "calls": [], "cost_usd": 0.5}]
    c1 += [_row(c, i < 10) for i, c in enumerate(ids)]
    c2 = [_row(c, i < 12) for i, c in enumerate(ids)]
    s1 = [_row(c, True, cost=0.01, route=[STRONG], latency=3000) for c in ids]
    s1.append(_row("narr", True, kind="narrative"))
    paths = [_write(tmp_path, "c1", cheap, c1), _write(tmp_path, "c2", cheap, c2),
             _write(tmp_path, "s1", strong, s1)]

    out = router.score(paths, reference=cheap, n_boot=500)

    assert out["numeric_cases_common"] == 20
    a = out["arms"][cheap]
    assert a["numeric_common"]["correct_by_run"] == [10, 12]
    assert a["numeric_common"]["accuracy_mean"] == pytest.approx(0.55)
    assert a["numeric_common"]["flips_run1_to_run2"] == {"fail_to_pass": 2, "pass_to_fail": 0}
    lo, hi = a["numeric_common"]["ci95"]
    assert lo < 0.55 < hi
    assert out["largest_same_arm_spread_cases"] == 2

    run1 = a["runs"][0]
    assert run1["errors"] == 0 and run1["numeric"]["correct"] == 10
    assert run1["cost_usd"]["per_1k_questions"] == pytest.approx(1.0)
    assert run1["cost_usd"]["per_1k_correct_numeric"] == pytest.approx(2.0)
    assert run1["cost_usd"]["spent_including_retried_rows"] == pytest.approx(0.52)

    s = out["arms"][strong]
    assert s["runs"][0]["narrative_uncalibrated_judge"] == {"n": 1, "passed": 1}
    assert s["runs"][0]["latency_ms"] == {"p50": 3000, "p95": 3000}
    vs = s["vs_reference"]
    assert vs["diff_cases"] == pytest.approx(9.0)
    assert vs["ci95"][0] > 0 and vs["claimed"] is True


def test_score_does_not_claim_a_gap_inside_the_same_arm_spread(tmp_path):
    ids = [f"n{i}" for i in range(40)]
    a, b = f"single:{CHEAP}", f"single:{STRONG}"
    paths = [
        _write(tmp_path, "a1", a, [_row(c, i < 20) for i, c in enumerate(ids)]),
        _write(tmp_path, "a2", a, [_row(c, i < 30) for i, c in enumerate(ids)]),
        _write(tmp_path, "b1", b, [_row(c, i < 33) for i, c in enumerate(ids)]),
    ]
    vs = router.score(paths, reference=a, n_boot=500)["arms"][b]["vs_reference"]
    # +8 cases over a's mean, interval clear of zero, but a's own runs differ by 10
    assert vs["diff_cases"] == pytest.approx(8.0) and vs["ci95"][0] > 0
    assert vs["claimed"] is False


def test_score_rejects_an_unknown_reference(tmp_path):
    path = _write(tmp_path, "a1", f"single:{CHEAP}", [_row("n0", True)])
    with pytest.raises(ValueError, match="reference"):
        router.score([path], reference=f"single:{STRONG}")
