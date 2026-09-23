"""Grading, judge calibration, extraction scoring, and the fake providers."""

from __future__ import annotations

import pytest

from edgar_intel.evals.judge import KAPPA_FLOOR, compute_kappa, grade_numeric, kappa_verdict, parse_number
from edgar_intel.evals.schemas import EvalCase
from edgar_intel.finetune.dataset import parse_extraction, score_extraction
from edgar_intel.providers.fake import FakeEmbedder, FakeLLM, FakeReranker


def numeric_case(value: float, unit: str = "USD") -> EvalCase:
    return EvalCase(
        case_id="t", kind="numeric", question="q",
        expected=str(value), expected_value=value, unit=unit,
    )


class TestParseNumber:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("$383,285", 383285),
            ("383285", 383285),
            ("$383.29 billion", 383_290_000_000),
            ("1.5 million", 1_500_000),
            ("12 thousand", 12_000),
            ("-4.2", -4.2),
        ],
    )
    def test_parses_magnitudes(self, text, expected):
        assert parse_number(text) == pytest.approx(expected, rel=1e-6)

    def test_returns_none_without_a_number(self):
        assert parse_number("no figures disclosed") is None


class TestNumericGrading:
    def test_exact_match(self):
        passed, score, _ = grade_numeric(numeric_case(383_285_000_000), "$383,285,000,000")
        assert passed and score == 1.0

    def test_accepts_filing_scale_in_millions(self):
        """Filings print 'in millions'. The model answering 383,285 is right.

        Without scale tolerance this reads as a factor-of-a-million error and
        the eval reports a false failure -- which then gets debugged as a model
        problem for a day.
        """
        passed, _, why = grade_numeric(numeric_case(383_285_000_000), "383,285")
        assert passed
        assert "millions" in why

    def test_rejects_a_genuinely_wrong_number(self):
        passed, score, _ = grade_numeric(numeric_case(383_285_000_000), "$120,000,000,000")
        assert not passed and score == 0.0

    def test_within_tolerance(self):
        passed, _, _ = grade_numeric(numeric_case(1_000_000), "1,001,000")
        assert passed

    def test_outside_tolerance(self):
        passed, _, _ = grade_numeric(numeric_case(1_000_000), "1,100,000")
        assert not passed

    def test_percent_cases_use_absolute_difference(self):
        case = numeric_case(-2.8, unit="percent")
        assert grade_numeric(case, "decreased 3.0%")[0]
        assert not grade_numeric(case, "decreased 12%")[0]

    def test_direction_word_carries_the_sign(self):
        """The minus sign lives in the word, not the number.

        "decreased 3.0%" is -3.0. Reading the bare figure would mark a correct
        answer wrong -- and, far worse, would mark "increased 2.8%" correct
        against an expected -2.8, passing an answer that says the opposite of
        the truth.
        """
        declined = numeric_case(-2.8, unit="percent")
        assert grade_numeric(declined, "Revenue fell 2.8% year over year.")[0]
        assert not grade_numeric(declined, "Revenue increased 2.8% year over year.")[0]

        grew = numeric_case(5.4, unit="percent")
        assert grade_numeric(grew, "Revenue rose 5.4%.")[0]
        assert not grade_numeric(grew, "Revenue declined 5.4%.")[0]

    def test_explicit_sign_beats_direction_word(self):
        case = numeric_case(-2.8, unit="percent")
        assert grade_numeric(case, "the change was -2.8%")[0]

    def test_no_number_in_answer_fails_with_a_reason(self):
        passed, _, why = grade_numeric(numeric_case(100_000), "The filing does not say.")
        assert not passed
        assert "no number" in why.lower()

    def test_missing_ground_truth_fails_loudly(self):
        case = EvalCase(case_id="t", kind="numeric", question="q", expected="?")
        passed, _, why = grade_numeric(case, "123456")
        assert not passed
        assert "expected_value" in why

    def test_percent_grading_ignores_underlying_currency_figures(self):
        """Regression: dollar values must not win over the YoY percentage."""
    case = numeric_case(-3.36, unit="percent")
    answer = (
        "Revenue fell from $67,060 million to $64,809 million, "
        "a decline of 3.4%."
    )

    passed, score, why = grade_numeric(case, answer)

    assert passed
    assert score == 1.0
    assert "read -3.40%" in why


class TestKappaGating:
    def test_too_few_labels_returns_none(self):
        assert compute_kappa([(True, True)] * 5) is None

    def test_enough_labels_computes(self):
        assert compute_kappa([(True, True)] * 25) == 1.0

    def test_verdict_flags_an_untrustworthy_judge(self):
        message = kappa_verdict(0.20)
        assert "BELOW" in message
        assert "do not report" in message

    def test_verdict_accepts_a_good_judge(self):
        assert "usable" in kappa_verdict(0.75)

    def test_verdict_explains_missing_calibration(self):
        """Two reasons a run has no kappa, and they mean different things.

        No overlap at all is the normal case -- labels attach to a specific
        answer hash, so a run that generated new answers has none of them. That
        is a category statement, not an outstanding task, and saying
        "uncalibrated" there reads as a to-do.
        """
        assert "not applicable to this run" in kappa_verdict(None, n_labels_matched=0)
        assert "uncalibrated" in kappa_verdict(None, n_labels_matched=7)

    def test_floor_is_the_conventional_substantial_agreement_threshold(self):
        assert KAPPA_FLOOR == 0.60


class TestExtractionScoring:
    def test_perfect_extraction(self):
        facts = [{"tag": "Revenues", "fiscal_year": 2023, "value": 100.0, "unit": "USD"}]
        score = score_extraction(facts, facts)
        assert score["f1"] == 1.0
        assert score["exact_set_match"] == 1.0

    def test_missing_fact_lowers_recall_only(self):
        expected = [
            {"tag": "Revenues", "fiscal_year": 2023, "value": 100.0},
            {"tag": "NetIncomeLoss", "fiscal_year": 2023, "value": 20.0},
        ]
        predicted = expected[:1]
        score = score_extraction(predicted, expected)
        assert score["precision"] == 1.0
        assert score["recall"] == 0.5

    def test_hallucinated_fact_lowers_precision_only(self):
        expected = [{"tag": "Revenues", "fiscal_year": 2023, "value": 100.0}]
        predicted = expected + [{"tag": "Assets", "fiscal_year": 2023, "value": 50.0}]
        score = score_extraction(predicted, expected)
        assert score["recall"] == 1.0
        assert score["precision"] == 0.5

    def test_empty_prediction_on_empty_expected_is_correct(self):
        """Negatives matter: a passage with no facts should extract nothing."""
        score = score_extraction([], [])
        assert score["f1"] == 1.0

    def test_value_tolerance(self):
        expected = [{"tag": "Revenues", "fiscal_year": 2023, "value": 1_000_000.0}]
        predicted = [{"tag": "Revenues", "fiscal_year": 2023, "value": 1_001_000.0}]
        assert score_extraction(predicted, expected)["f1"] == 1.0

    def test_parses_json_surrounded_by_prose(self):
        text = 'Here you go:\n[{"tag":"Revenues","fiscal_year":2023,"value":1}]\nHope that helps.'
        assert parse_extraction(text)[0]["tag"] == "Revenues"

    def test_unparseable_returns_empty(self):
        assert parse_extraction("no json here") == []


class TestFakeProviders:
    def test_embedder_is_deterministic(self):
        e = FakeEmbedder(dim=64)
        assert e.embed(["hello world"]) == e.embed(["hello world"])

    def test_embedder_carries_real_lexical_signal(self):
        """A fake that returns noise makes the CI eval gate meaningless."""
        e = FakeEmbedder(dim=256)
        a, b, c = e.embed([
            "revenue increased due to higher volumes",
            "revenue growth driven by volume increases",
            "the board approved a new dividend policy",
        ])
        sim = lambda x, y: sum(i * j for i, j in zip(x, y, strict=True))  # noqa: E731
        assert sim(a, b) > sim(a, c)

    def test_embedder_is_normalised(self):
        vec = FakeEmbedder(dim=128).embed(["some text here"])[0]
        assert sum(v * v for v in vec) == pytest.approx(1.0, abs=1e-6)

    def test_llm_extracts_from_context(self):
        llm = FakeLLM()
        out = llm.complete("CONTEXT:\nTotal net sales 383,285\n\nQUESTION:\nWhat were sales?")
        assert "383285" in out.text.replace(",", "")

    def test_llm_judge_agrees_on_matching_answers(self):
        llm = FakeLLM()
        prompt = (
            "QUESTION:\nWhat risk?\n\nREFERENCE:\nsupplier concentration risk\n\n"
            "CANDIDATE:\nthe company discloses supplier concentration risk\n"
        )
        out = llm.complete(prompt, json_schema={
            "properties": {"verdict": {"type": "boolean"}, "rationale": {"type": "string"}}
        })
        assert '"verdict": true' in out.text

    def test_llm_judge_disagrees_on_unrelated_answers(self):
        llm = FakeLLM()
        prompt = (
            "QUESTION:\nWhat risk?\n\nREFERENCE:\nsupplier concentration risk\n\n"
            "CANDIDATE:\nthe weather was pleasant in June\n"
        )
        out = llm.complete(prompt, json_schema={
            "properties": {"verdict": {"type": "boolean"}, "rationale": {"type": "string"}}
        })
        assert '"verdict": false' in out.text

    def test_reranker_prefers_overlapping_passages(self):
        scores = FakeReranker().score(
            "supplier concentration risk",
            ["supplier concentration is a risk", "dividends were declared quarterly"],
        )
        assert scores[0] > scores[1]

    def test_llm_reports_token_counts(self):
        out = FakeLLM().complete("a fairly long prompt " * 20)
        assert out.prompt_tokens > 0
        assert out.total_tokens == out.prompt_tokens + out.completion_tokens


class TestGoldenSetValidity:
    """The generated set must only ask questions the corpus can answer.

    `companyfacts` returns a company's entire XBRL history; ingestion fetches
    only the most recent filings. Generating a question for a year whose filing
    was never ingested produces a case retrieval cannot answer at any quality
    level -- and on the first real run that was 185 of 208 numeric cases, every
    one of them scoring zero for reasons that had nothing to do with retrieval.
    """

    def test_keeps_only_covered_periods(self):
        from edgar_intel.evals.goldenset import filter_to_covered

        facts = [
            {"cik": "0001", "fiscal_year": 2024, "tag": "Revenues"},
            {"cik": "0001", "fiscal_year": 2011, "tag": "Revenues"},
            {"cik": "0002", "fiscal_year": 2024, "tag": "Revenues"},
        ]
        covered = {("0001", 2024)}
        kept = filter_to_covered(facts, covered)
        assert len(kept) == 1
        assert kept[0]["fiscal_year"] == 2024
        assert kept[0]["cik"] == "0001"

    def test_coverage_is_per_company_not_global(self):
        """FY2024 being ingested for one company says nothing about another."""
        from edgar_intel.evals.goldenset import filter_to_covered

        facts = [
            {"cik": "0001", "fiscal_year": 2024},
            {"cik": "0002", "fiscal_year": 2024},
        ]
        kept = filter_to_covered(facts, {("0001", 2024)})
        assert [f["cik"] for f in kept] == ["0001"]

    def test_empty_coverage_yields_nothing(self):
        from edgar_intel.evals.goldenset import filter_to_covered

        assert filter_to_covered([{"cik": "0001", "fiscal_year": 2024}], set()) == []

    def test_fiscal_year_type_mismatch_is_tolerated(self):
        """Postgres may hand back the year as a str depending on the driver."""
        from edgar_intel.evals.goldenset import filter_to_covered

        assert len(filter_to_covered([{"cik": "0001", "fiscal_year": "2024"}], {("0001", 2024)})) == 1


class TestNoteFormatting:
    def test_large_values_lose_the_decimals(self):
        from edgar_intel.evals.goldenset import _fmt

        assert _fmt(383_285_000_000) == "383,285,000,000"

    def test_per_share_values_keep_them(self):
        """A fixed '%,.0f' rendered EPS of 1.43 as "1", so a note read
        "FY2009=1, FY2010=1" next to "decreased 55.2%" -- visibly nonsense."""
        from edgar_intel.evals.goldenset import _fmt

        assert _fmt(1.43) == "1.43"
        assert _fmt(6.11) == "6.11"

    def test_negative_values(self):
        from edgar_intel.evals.goldenset import _fmt

        assert _fmt(-2500) == "-2,500"


class TestFakeLLMAnswersFromContext:
    """The fake provider is what makes the CI eval gate meaningful.

    It previously took the first number in the context, which is the passage
    marker `build_context` prefixes each passage with -- so every numeric answer
    came back as "1" and the gate scored 0% whether or not retrieval worked. A
    gate that always fails measures nothing.
    """

    def test_ignores_the_citation_marker(self):
        from edgar_intel.providers.fake import FakeLLM

        context = "CONTEXT:\n[1] AAPL FY2023 Item 7\nTotal net sales | 383,285 | 394,328\n"
        answer = FakeLLM().complete(context + "\nQUESTION:\nWhat were net sales?").text
        assert answer != "1"
        assert "383285" in answer.replace(",", "") or "394328" in answer.replace(",", "")

    def test_ignores_fiscal_year_labels(self):
        from edgar_intel.providers.fake import FakeLLM

        context = "CONTEXT:\n[2] MSFT FY2024 Item 7\nResearch and development | 29,510\n"
        answer = FakeLLM().complete(context + "\nQUESTION:\nWhat was R&D?").text
        assert "2024" not in answer
        assert "29510" in answer.replace(",", "")

    def test_grades_correct_against_the_real_expected_value(self):
        """End to end: the gate must be able to score a numeric case as passing."""
        from edgar_intel.evals.judge import grade_numeric
        from edgar_intel.providers.fake import FakeLLM

        context = "CONTEXT:\n[1] CAT FY2025 Item 7\nCash and cash equivalents | 6,889\n"
        answer = FakeLLM().complete(context + "\nQUESTION:\nHow much cash?").text
        passed, _, why = grade_numeric(numeric_case(6_889_000_000), answer)
        assert passed, why
