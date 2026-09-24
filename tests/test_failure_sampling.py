"""The failure sample was three companies out of eight, and said nothing about it.

D1 was a global `[:50]` cap that let numeric failures crowd out all 24 narrative
ones, so every report showed `narrative_pass_rate: 0.0` with no example of why.
The fix gave each kind its own quota of 25. It was the right fix at the wrong
level, and it left the identical defect one step down.

Taking the *first* 25 of a kind is not a sample. Results come out in CIK order,
so on `baseline-v5-2943b37a` the 25 numeric rows were CAT, PG and JNJ -- the
three lowest CIKs -- and AAPL, UNH, MSFT, COST and NVDA did not appear at all.
54 of 79 numeric failures were missing from the file, and the D3 analysis of
"what is still failing" was written from three companies while reading as though
it covered eight.

Two of the four per-case figures carried in the evidence log could not be
confirmed against the report for exactly this reason. That is what a biased
sample costs: not a wrong number, a number you cannot check.
"""

from __future__ import annotations

from edgar_intel.evals.runner import (
    FAILURE_SAMPLE_PER_KIND,
    failure_counts_by_kind,
    failures_omitted_by_kind,
    sample_failures_by_kind,
    subject_of,
)
from edgar_intel.evals.schemas import CaseResult

# CIK order, which is the order results arrive in.
TICKERS = ["CAT", "PG", "JNJ", "AAPL", "UNH", "MSFT", "COST", "NVDA"]


def _fail(case_id: str, kind: str = "numeric") -> CaseResult:
    return CaseResult(
        case_id=case_id,
        kind=kind,
        passed=False,
        score=0.0,
        answer="a",
        expected="b",
        judge_rationale="why",
    )


def _pass(case_id: str, kind: str = "numeric") -> CaseResult:
    r = _fail(case_id, kind)
    r.passed = True
    r.score = 1.0
    return r


def _baseline_v5_shape() -> list[CaseResult]:
    """79 numeric failures in CIK order, ten per company, plus 5 narrative."""
    out: list[CaseResult] = []
    for t in TICKERS:
        for i in range(10):
            out.append(_fail(f"num-{t}-Concept{i}-2025"))
    out = out[:79]
    for t in TICKERS[:5]:
        out.append(_fail(f"nar-{t}-legal", kind="narrative"))
    return out


class TestSubjectOf:
    def test_reads_the_ticker_from_the_id(self):
        assert subject_of("num-PG-NetIncomeLoss-2025") == "PG"
        assert subject_of("yoy-CAT-ResearchAndDevelopmentExpense-2023-2024") == "CAT"
        assert subject_of("nar-AAPL-supply-concentration") == "AAPL"

    def test_an_unrecognised_shape_shares_one_bucket_rather_than_vanishing(self):
        assert subject_of("weird") == "?"
        assert subject_of("num-lowercase-thing") == "?"
        assert subject_of("") == "?"


class TestTheQuotaSpreadsAcrossCompanies:
    def test_every_company_with_a_failure_is_represented(self):
        """The actual regression: five of eight companies were invisible."""
        rows = sample_failures_by_kind(_baseline_v5_shape())
        numeric = [r for r in rows if r["kind"] == "numeric"]
        assert len(numeric) == FAILURE_SAMPLE_PER_KIND
        assert {r["ticker"] for r in numeric} == set(TICKERS)

    def test_the_old_behaviour_would_have_failed_this(self):
        """Documents what was wrong: first-25-in-order covers three tickers."""
        failures = [r for r in _baseline_v5_shape() if not r.passed and r.kind == "numeric"]
        first_25 = {subject_of(r.case_id) for r in failures[:25]}
        assert len(first_25) == 3
        assert first_25 == {"CAT", "PG", "JNJ"}

    def test_a_company_with_one_failure_is_shown_before_a_noisy_one_repeats(self):
        results = [_fail(f"num-PG-C{i}-2025") for i in range(30)]
        results.append(_fail("num-NVDA-Only-2025"))
        rows = sample_failures_by_kind(results)
        assert "NVDA" in {r["ticker"] for r in rows}

    def test_the_quota_is_still_honoured(self):
        rows = sample_failures_by_kind(_baseline_v5_shape())
        for kind in ("numeric", "narrative"):
            assert len([r for r in rows if r["kind"] == kind]) <= FAILURE_SAMPLE_PER_KIND

    def test_fewer_failures_than_the_quota_returns_all_of_them(self):
        results = [_fail(f"num-{t}-C0-2025") for t in TICKERS[:3]]
        assert len(sample_failures_by_kind(results)) == 3

    def test_passes_are_never_sampled(self):
        results = [_pass(f"num-{t}-C0-2025") for t in TICKERS]
        assert sample_failures_by_kind(results) == []

    def test_one_company_only_still_fills_the_quota(self):
        """Round-robin must not starve the sample when there is nothing to spread."""
        results = [_fail(f"num-PG-C{i}-2025") for i in range(40)]
        assert len(sample_failures_by_kind(results)) == FAILURE_SAMPLE_PER_KIND


class TestNarrativeIsStillNotCrowdedOut:
    """D1 must not come back while fixing its descendant."""

    def test_narrative_survives_seventy_nine_numeric_failures(self):
        rows = sample_failures_by_kind(_baseline_v5_shape())
        narrative = [r for r in rows if r["kind"] == "narrative"]
        assert len(narrative) == 5

    def test_kinds_are_grouped_and_ordered(self):
        rows = sample_failures_by_kind(_baseline_v5_shape())
        kinds = [r["kind"] for r in rows]
        assert kinds == sorted(kinds)


class TestTheReportSaysWhatItIsNotShowing:
    def test_omitted_counts_are_reported_per_kind(self):
        results = _baseline_v5_shape()
        assert failure_counts_by_kind(results) == {"numeric": 79, "narrative": 5}
        assert failures_omitted_by_kind(results) == {"numeric": 54, "narrative": 0}

    def test_nothing_omitted_reads_as_zero_not_absent(self):
        results = [_fail(f"num-{t}-C0-2025") for t in TICKERS[:3]]
        assert failures_omitted_by_kind(results) == {"numeric": 0}

    def test_the_omitted_count_matches_the_sample_it_describes(self):
        results = _baseline_v5_shape()
        rows = sample_failures_by_kind(results)
        counts = failure_counts_by_kind(results)
        omitted = failures_omitted_by_kind(results)
        for kind, total in counts.items():
            shown = len([r for r in rows if r["kind"] == kind])
            assert shown + omitted[kind] == total

    def test_the_report_payload_carries_it(self):
        """Otherwise the reader subtracts two numbers, and nobody subtracts."""
        import ast
        import pathlib

        src = pathlib.Path(
            "src/edgar_intel/evals/runner.py"
        ).read_text()
        tree = ast.parse(src)
        write = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_write_report"
        )
        keys = {
            k.value
            for n in ast.walk(write)
            if isinstance(n, ast.Dict)
            for k in n.keys
            if isinstance(k, ast.Constant)
        }
        assert "failures_omitted_by_kind" in keys
        assert "failure_counts" in keys
