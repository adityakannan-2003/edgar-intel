"""Number extraction and abstention detection.

Every string in this file is copied from `reports/smoke20-011be57c.json`. That
run reported 25% numeric accuracy. Fifteen of the twenty cases were recorded as
wrong; on inspection the grader was at fault in most of them, reading the
fiscal year out of the question's own echo and calling it the model's answer.

A grader bug is worse than a model bug. A model bug shows up as a bad number you
go and investigate. A grader bug shows up as a bad number that makes you
investigate the model.
"""

from __future__ import annotations

import pytest

from edgar_intel.evals.judge import is_abstention, parse_number


class TestTheYearIsNotTheAnswer:
    def test_fy_label_is_not_read_as_the_figure(self):
        """Graded as `got 2,025.00` against an expected 2,148,000,000."""
        answer = "The research and development expense for Caterpillar Inc. in FY2025 was $2,148 million."
        assert parse_number(answer) == pytest.approx(2_148_000_000)

    def test_fiscal_year_spelled_out(self):
        answer = "For fiscal year 2023, total revenue was $67,060 million."
        assert parse_number(answer) == pytest.approx(67_060_000_000)

    def test_period_ended_phrasing(self):
        answer = "For the year ended December 31, 2024, net income was $10.79 billion."
        assert parse_number(answer) == pytest.approx(10_790_000_000)

    def test_a_bare_year_alone_yields_no_number(self):
        """Better to report 'no number found' than to invent one from the date."""
        assert parse_number("The figure was reported in 2024.") is None

    def test_an_iso_date_is_not_a_figure(self):
        assert parse_number("As of 2024-12-31 the balance was $6,978 million.") == pytest.approx(
            6_978_000_000
        )


class TestScaleAndSignal:
    def test_prefers_a_currency_figure_over_incidental_digits(self):
        answer = "Across 3 segments and 2 regions, revenue was $64,809 million."
        assert parse_number(answer) == pytest.approx(64_809_000_000)

    def test_scale_words(self):
        assert parse_number("$383.29 billion") == pytest.approx(383_290_000_000)
        assert parse_number("$2,148 million") == pytest.approx(2_148_000_000)

    def test_plain_thousands_separated_figure(self):
        """Filings print 'in millions'; the grader rescales, not the parser."""
        assert parse_number("67,973") == pytest.approx(67_973)

    def test_per_share_figures_are_not_treated_as_years(self):
        """$20.12 is four significant digits and must not be mistaken for a date."""
        assert parse_number("Diluted EPS was $20.12 for the period.") == pytest.approx(20.12)

    def test_percentages(self):
        assert parse_number("Revenue decreased 3.4% year over year.") == pytest.approx(3.4)

    def test_negative(self):
        assert parse_number("-$1,200 million") == pytest.approx(-1_200_000_000)

    def test_no_digits_at_all(self):
        assert parse_number("Revenue grew modestly.") is None


class TestAbstention:
    """The eight failures that were not failures of the model.

    The system prompt asks the model to say when the context lacks the answer.
    It did. The grader parsed a year out of the refusal and recorded a wrong
    figure, which made a retrieval problem look like a hallucination problem.
    """

    @pytest.mark.parametrize(
        "answer",
        [
            "The context does not provide information on the amount of cash and "
            "cash equivalents reported by Caterpillar Inc. in FY2025.",
            "The context does not provide specific information about Caterpillar "
            "Inc.'s research and development expense for fiscal year 2023.",
            "The provided passages do not contain the requested figure.",
            "This information is not available in the supplied context.",
            "The diluted earnings per share is not stated in the context.",
            "I cannot determine the total liabilities from the passages provided.",
        ],
    )
    def test_recognised(self, answer):
        assert is_abstention(answer)

    @pytest.mark.parametrize(
        "answer",
        [
            "Caterpillar Inc. reported approximately 465 million common shares outstanding.",
            "$22.05",
            "Total revenue was $67,589 million, up from the prior year.",
            "The company does not break out the figure by segment, but total "
            "research and development expense was $2,148 million.",
        ],
    )
    def test_a_real_answer_is_not_an_abstention(self, answer):
        assert not is_abstention(answer)

    def test_abstention_short_circuits_numeric_grading(self):
        """Otherwise the refusal's own words get parsed as the answer."""
        from edgar_intel.evals.judge import grade_numeric
        from edgar_intel.evals.schemas import EvalCase

        case = EvalCase(
            case_id="num-CAT-Revenues-2025",
            kind="numeric",
            question="What was Caterpillar's total revenue for fiscal year 2025?",
            expected="$64.81 billion",
            expected_value=64_809_000_000,
            unit="USD",
        )
        passed, score, why = grade_numeric(
            case, "The context does not provide Caterpillar's FY2025 revenue."
        )
        assert not passed
        assert score == 0.0
        assert why.startswith("ABSTAINED")

    def test_a_correct_answer_still_passes_after_the_change(self):
        from edgar_intel.evals.judge import grade_numeric
        from edgar_intel.evals.schemas import EvalCase

        case = EvalCase(
            case_id="num-CAT-ResearchAndDevelopmentExpense-2025",
            kind="numeric",
            question="What was Caterpillar's R&D expense in FY2025?",
            expected="$2.15 billion",
            expected_value=2_148_000_000,
            unit="USD",
        )
        passed, score, why = grade_numeric(
            case,
            "The research and development expense for Caterpillar Inc. in FY2025 "
            "was $2,148 million.",
        )
        assert passed, why
        assert score == 1.0


class TestInfrastructureFailuresAreNotScores:
    """The 232-case run that reported an overall score of 0.0.

    Every case failed with `401 Unauthorized`. The harness recorded the run,
    wrote `reports/baseline-45536dc9.json`, and put a zero in the metrics table
    that described a missing environment variable rather than the system.
    """

    def test_auth_failure_is_recognised(self):
        from edgar_intel.evals.runner import is_infra_error

        assert is_infra_error(
            "Client error '401 Unauthorized' for url "
            "'https://api.openai.com/v1/chat/completions'"
        )

    @pytest.mark.parametrize(
        "message",
        [
            "429 Too Many Requests",
            "Connection refused",
            "insufficient_quota",
            "Read Timeout",
        ],
    )
    def test_other_infrastructure_failures(self, message):
        from edgar_intel.evals.runner import is_infra_error

        assert is_infra_error(message)

    @pytest.mark.parametrize(
        "message",
        ["", "no number found in the answer", "expected 20.12, got 22.05"],
    )
    def test_a_grading_failure_is_not_an_infrastructure_failure(self, message):
        from edgar_intel.evals.runner import is_infra_error

        assert not is_infra_error(message)
