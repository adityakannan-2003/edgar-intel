"""D2: comparative cases failed answers that were right.

"How did X change from FY2023 to FY2024?" does not ask for a percentage, and the
model mostly answers the way a filing does -- both figures, a direction, a
dollar delta. The grader only looked for a percentage, so
`yoy-CAT-ResearchAndDevelopmentExpense-2023-2024` ("FY2024 was $2,107 million, a
decrease from $2,108 million in FY2023") failed for "no percentage found".

It was not one case. Of the 11 comparative failures in `baseline-v5`'s failure
sample, 8 were this shape; the other three are two abstentions and the D3
concept case. Every answer string below is verbatim from
`reports/baseline-v5-2943b37a.json`, and every expected value and source-fact
note from `evalset/golden.json`.

The fallback still has to be earned: two *different* figures matching the two
XBRL values, an implied change inside the same 0.5-point tolerance a stated
percentage gets, and no self-contradiction (a direction word opposite to the
change, or the years attached the wrong way round). And a stated percentage
still governs when there is one.
"""

from __future__ import annotations

import pytest

from edgar_intel.evals.judge import grade_numeric, parse_percent
from edgar_intel.evals.schemas import EvalCase


def _yoy(case_id: str, expected: float, notes: str) -> EvalCase:
    return EvalCase(
        case_id=case_id, kind="numeric", question="How did it change?",
        expected=f"{expected:+.1f}%", expected_value=expected, unit="percent",
        difficulty="comparative", notes=notes,
    )


BASELINE_V5_FALSE_FAILURES = [
    (
        "yoy-CAT-Revenues-2023-2024", -3.3567,
        "FY2023=67,060,000,000, FY2024=64,809,000,000",
        "Total sales and revenues for 2024 were $64.809 billion, a decrease of $2.251 "
        "billion, or 3 percent, compared with $67.060 billion in 2023.",
    ),
    (
        "yoy-CAT-ResearchAndDevelopmentExpense-2023-2024", -0.0474,
        "FY2023=2,108,000,000, FY2024=2,107,000,000",
        "Caterpillar Inc.'s research and development expense for FY2024 was $2,107 "
        "million, which is a decrease from $2,108 million in FY2023.",
    ),
    (
        "yoy-CAT-EarningsPerShareDiluted-2023-2024", 9.5924,
        "FY2023=20.12, FY2024=22.05",
        "Caterpillar Inc.'s diluted earnings per share increased from $20.12 in FY2023 "
        "to $22.05 in FY2024.",
    ),
    (
        "yoy-CAT-OperatingIncomeLoss-2023-2024", 0.8175,
        "FY2023=12,966,000,000, FY2024=13,072,000,000",
        "The operating income for FY2024 was $13,072 million and for FY2023 was $12,966 "
        "million. The year-over-year change in operating income is an increase of $106 "
        "million.",
    ),
    (
        "yoy-CAT-Liabilities-2023-2024", 0.4369,
        "FY2023=67,973,000,000, FY2024=68,270,000,000",
        "The total liabilities for FY2023 were $67,973 million and for FY2024 were "
        "$68,270 million. The year-over-year change in total liabilities is an increase "
        "of $297 million.",
    ),
    (
        "yoy-PG-Liabilities-2024-2025", 1.5805,
        "FY2024=71,811,000,000, FY2025=72,946,000,000",
        "Procter & Gamble Co's total liabilities increased from $71,811 million in "
        "FY2024 to $72,946 million in FY2025, reflecting a change of $1,135 million.",
    ),
    (
        "yoy-PG-OperatingIncomeLoss-2024-2025", 10.2777,
        "FY2024=18,545,000,000, FY2025=20,451,000,000",
        "Operating income increased from $18.545 billion in FY2024 to $20.451 billion in "
        "FY2025, which is an increase of $1.906 billion.",
    ),
    (
        "yoy-PG-Assets-2024-2025", 2.338,
        "FY2024=122,370,000,000, FY2025=125,231,000,000",
        "The total assets for Procter & Gamble Co. in FY2024 were $122,370 million, and "
        "in FY2025 they were $125,231 million. The year-over-year change in total "
        "assets is $125,231 million - $122,370 million = $2,861 million.",
    ),
]


class TestTheBaselineV5AnswersThatWereRight:
    @pytest.mark.parametrize(
        "case_id, expected, notes, answer",
        BASELINE_V5_FALSE_FAILURES,
        ids=[row[0] for row in BASELINE_V5_FALSE_FAILURES],
    )
    def test_now_pass(self, case_id, expected, notes, answer):
        passed, score, why = grade_numeric(_yoy(case_id, expected, notes), answer)
        assert passed, why
        assert score == 1.0

    def test_the_rationale_says_which_path_passed_it(self):
        case_id, expected, notes, answer = BASELINE_V5_FALSE_FAILURES[1]
        _, _, why = grade_numeric(_yoy(case_id, expected, notes), answer)
        assert why.startswith("no percentage stated; both figures match")
        assert "FY2023 2,108,000,000" in why and "FY2024 2,107,000,000" in why


class TestTheOnesThatWereWrongStayWrong:
    def test_the_d3_concept_case_still_fails(self):
        """A stated percentage governs. PG net income is a D3 problem, not D2."""
        case = _yoy(
            "yoy-PG-NetIncomeLoss-2024-2025", 7.3594,
            "FY2024=14,879,000,000, FY2025=15,974,000,000",
        )
        answer = (
            "Procter & Gamble's net income increased from $14.879 billion in FY2024 to "
            "$16.065 billion in FY2025, which is a year-over-year change of $1.186 "
            "billion, or approximately 8%."
        )
        passed, _, why = grade_numeric(case, answer)
        assert not passed
        assert "read +8.00%" in why

    def test_abstentions_still_fail_as_abstentions(self):
        case = _yoy("yoy-PG-ResearchAndDevelopmentExpense-2024-2025", 5.0, "")
        passed, _, why = grade_numeric(
            case,
            "The context does not provide information about Procter & Gamble Co's "
            "research and development expense for FY2024 or FY2025, so I cannot "
            "determine how it changed.",
        )
        assert not passed and why.startswith("ABSTAINED")


class TestTheFallbackHasToBeEarned:
    NEAR_FLAT = ("yoy-CAT-R&D", -0.0474, "FY2023=2,108,000,000, FY2024=2,107,000,000")
    REVENUE = ("yoy-CAT-Rev", -3.3567, "FY2023=67,060,000,000, FY2024=64,809,000,000")

    def test_one_figure_is_not_two(self):
        """2,107 is within 0.5% of both 2,107 and 2,108 -- one figure must not
        count as both."""
        passed, _, why = grade_numeric(
            _yoy(*self.NEAR_FLAT), "R&D expense was $2,107 million in FY2024."
        )
        assert not passed
        assert "does not state both the FY2023 and FY2024 figures" in why

    def test_the_old_message_survives_when_nothing_matches(self):
        passed, _, why = grade_numeric(
            _yoy(*self.REVENUE), "Revenue went down by about $2 billion."
        )
        assert not passed and why.startswith("no percentage found in the answer")

    def test_a_case_without_source_facts_fails_as_before(self):
        passed, _, why = grade_numeric(
            _yoy("yoy-x", -3.36, ""), "Revenue fell from $67,060 million to $64,809 million."
        )
        assert not passed and why == "no percentage found in the answer"

    def test_the_wrong_direction_word_fails(self):
        passed, _, why = grade_numeric(
            _yoy(*self.REVENUE),
            "Revenue increased from $64,809 million in FY2023 to $67,060 million in FY2024.",
        )
        assert not passed
        assert "calls a -3.36% change an increase" in why

    def test_years_attached_the_wrong_way_round_fail(self):
        """The realistic misread: 10-K tables print the latest year first."""
        passed, _, why = grade_numeric(
            _yoy(*self.REVENUE), "FY2023: $64,809 million. FY2024: $67,060 million."
        )
        assert not passed
        assert "wrong years" in why

    def test_figures_each_in_tolerance_but_a_change_out_of_it_fail(self):
        """Each figure within 0.5% of its value; together they imply -0.99%, not
        -0.05%. The change gets the same 0.5-point tolerance as a percentage."""
        passed, _, why = grade_numeric(
            _yoy(*self.NEAR_FLAT),
            "It was $2,118 million in FY2023 and $2,097 million in FY2024.",
        )
        assert not passed
        assert "imply -0.99%" in why

    def test_a_wrong_stated_percentage_is_not_rescued_by_right_figures(self):
        passed, _, _ = grade_numeric(
            _yoy(*self.REVENUE),
            "Revenue fell from $67,060 million in FY2023 to $64,809 million in FY2024, "
            "a decline of 9%.",
        )
        assert not passed

    def test_a_tie_between_years_is_not_evidence_of_a_swap(self):
        """$22.05 sits four characters from both FY2023 and FY2024."""
        passed, _, why = grade_numeric(
            _yoy("yoy-EPS", 9.5924, "FY2023=20.12, FY2024=22.05"),
            "EPS went from $20.12 in FY2023 to $22.05 in FY2024.",
        )
        assert passed, why


class TestPercentWords:
    @pytest.mark.parametrize(
        "text, value",
        [
            ("a decline of 3.4%", 3.4),
            ("a decline of 3.4 %", 3.4),
            ("or 3 percent, compared with", 3.0),
            ("down 3.4 per cent", 3.4),
            ("up 12 pct", 12.0),
            ("the change was -2.8%", -2.8),
        ],
    )
    def test_the_word_counts_as_the_sign(self, text, value):
        assert parse_percent(text) == value

    def test_a_figure_is_not_a_percentage(self):
        assert parse_percent("Revenue was $64,809 million in FY2024.") is None

    def test_a_year_is_not_a_percentage(self):
        assert parse_percent("In fiscal 2024 percent-of-sales metrics improved") is None
