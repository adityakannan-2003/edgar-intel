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
        assert "uncalibrated" in kappa_verdict(None)

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
