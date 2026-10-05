"""Needles the evidence linker searches chunk text for (ILIKE '%needle%').

Every needle used to be `f"{scaled:,.0f}"`, so a per-share figure collapsed to
one or two digits: diluted EPS of 2.94 became "3", 11.93 became "12". The first
needle that hits wins, so each EPS case was labelled with whichever chunks of
that company's filings contained the digit -- and hit@k / recall@k for those
cases measured nothing.
"""

from __future__ import annotations

import pytest

from edgar_intel.evals.goldenset import _comparative_source_facts, _numeric_value_needles
from edgar_intel.evals.schemas import EvalCase

# NVDA FY2025 income statement, as chunked.
NVDA_EPS_ROW = "Diluted | $ | 2.94 | $ | 1.19 | $ | 0.17"


class TestPerShareValues:
    @pytest.mark.parametrize("value,needle", [(2.94, "2.94"), (1.19, "1.19"), (11.93, "11.93")])
    def test_printed_decimal_form(self, value, needle):
        assert _numeric_value_needles(value, "USD/shares") == [needle]

    @pytest.mark.parametrize("value,needle", [(2.94, "2.94"), (1.19, "1.19"), (11.93, "11.93")])
    def test_same_without_a_unit(self, value, needle):
        """YoY source values arrive with no unit: the case's own is "percent"."""
        assert _numeric_value_needles(value) == [needle]

    def test_keeps_the_trailing_zero_statements_print(self):
        """MSFT FY2024 prints "11.80"; the float is 11.8."""
        assert _numeric_value_needles(11.8, "USD/shares") == ["11.80"]

    def test_whole_dollar_eps(self):
        assert _numeric_value_needles(6.0, "USD/shares") == ["6.00"]
        assert _numeric_value_needles(6.0) == ["6.00"]

    def test_extra_precision_is_not_rounded_away(self):
        assert _numeric_value_needles(1.0065, "USD/shares") == ["1.0065"]

    def test_negative_matches_the_parenthesised_form(self):
        needles = _numeric_value_needles(-2.94, "USD/shares")
        assert needles == ["2.94"]
        assert needles[0] in "Diluted | $ | (2.94) | $ | 1.19"

    def test_large_per_share_value_keeps_its_decimals(self):
        assert _numeric_value_needles(1234.56, "USD/shares") == ["1,234.56"]

    def test_sub_dollar_value(self):
        assert _numeric_value_needles(0.17, "USD/shares") == ["0.17"]

    def test_finds_the_eps_row_and_not_a_stray_digit(self):
        assert all(n in NVDA_EPS_ROW for n in _numeric_value_needles(2.94, "USD/shares"))
        assert not any(
            n in "Item 3. Legal Proceedings" for n in _numeric_value_needles(2.94, "USD/shares")
        )


class TestLargeValues:
    def test_every_presentation_scale(self):
        assert _numeric_value_needles(383_285_000_000, "USD") == [
            "383,285,000,000", "383,285,000", "383,285", "383",
        ]

    def test_scale_that_leaves_too_few_digits_is_dropped(self):
        """2,108,000,000 in billions is "2"."""
        assert _numeric_value_needles(2_108_000_000, "USD") == [
            "2,108,000,000", "2,108,000", "2,108",
        ]

    def test_share_counts(self):
        assert _numeric_value_needles(15_812_547_000, "shares") == [
            "15,812,547,000", "15,812,547", "15,813",
        ]


class TestShortNeedles:
    @pytest.mark.parametrize(
        "value,unit",
        [
            (2.94, "USD/shares"), (0.17, "USD/shares"), (11.93, None), (6.0, None),
            (12.0, None), (1_500, "USD"), (2_108_000_000, "USD"), (12_500_000_000, None),
            (383_285_000_000, "USD"),
        ],
    )
    def test_no_needle_under_three_digits(self, value, unit):
        for needle in _numeric_value_needles(value, unit):
            assert sum(ch.isdigit() for ch in needle) >= 3, needle

    def test_zero_has_no_needle(self):
        assert _numeric_value_needles(0.0, "USD/shares") == []
        assert _numeric_value_needles(0.0) == []

    def test_small_whole_number_without_unit_falls_back_to_integer(self):
        assert _numeric_value_needles(500.0) == ["500.00", "500"]
        assert _numeric_value_needles(12.0) == ["12.00"]


class TestYoyNotes:
    def test_eps_notes_round_trip_to_decimal_needles(self):
        """The notes are rendered by `_fmt`, parsed back, then needled."""
        facts = _comparative_source_facts("FY2024=11.93, FY2025=2.94")
        assert [_numeric_value_needles(v) for _, v in facts] == [["11.93"], ["2.94"]]

    def test_usd_notes_unchanged(self):
        facts = _comparative_source_facts("FY2023=2,108,000,000, FY2024=2,107,000,000")
        assert _numeric_value_needles(facts[0][1])[0] == "2,108,000,000"


class TestLinkerWiring:
    """No database here: record the ILIKE patterns the linker sends."""

    @pytest.fixture
    def patterns(self, monkeypatch) -> list[str]:
        import edgar_intel.db as db

        seen: list[str] = []

        def query(sql, params=None):
            seen.extend(p for p in params if isinstance(p, str) and p.startswith("%"))
            return []

        monkeypatch.setattr(db, "query", query)
        return seen

    def test_single_hop_eps(self, patterns):
        from edgar_intel.evals.goldenset import link_evidence

        case = EvalCase(
            case_id="num-NVDA-EarningsPerShareDiluted-2024", kind="numeric", question="q?",
            expected="$11.93", expected_value=11.93, unit="USD/shares", ticker="NVDA",
            fiscal_year=2024, tag="EarningsPerShareDiluted",
        )
        link_evidence([case], "section_aware")
        assert patterns == ["%11.93%"]

    def test_yoy_eps(self, patterns):
        from edgar_intel.evals.goldenset import link_evidence

        case = EvalCase(
            case_id="yoy-NVDA-EarningsPerShareDiluted-2024-2025", kind="numeric",
            question="q?", expected="decreased 75.4%", expected_value=-75.3562,
            unit="percent", ticker="NVDA", fiscal_year=2025, tag="EarningsPerShareDiluted",
            difficulty="comparative", notes="FY2024=11.93, FY2025=2.94",
        )
        link_evidence([case], "section_aware")
        # Both values together, then each one alone.
        assert patterns == ["%11.93%", "%2.94%", "%11.93%", "%2.94%"]
