"""The fiscal-year attribution bug, and the guards against it returning.

This is the most damaging class of bug in the repo, because it corrupts the
*ground truth*. Everything downstream — the golden set, numeric accuracy, the
retrieval metrics computed against linked evidence — inherits the error, and
none of it raises. The system looks broken and the data is what is broken.

Every fixture below is shaped like the real `companyfacts` payload that caused
it: Caterpillar's FY2025 10-K, which reports three years of income statement and
stamps all three with `fy: 2025`.
"""

from __future__ import annotations

from datetime import date

from edgar_intel.ingest.xbrl import (
    conflicting_years,
    covers_full_year,
    extract_facts,
    fiscal_year_of,
)


def _payload(entries: list[dict], tag: str = "Revenues", unit: str = "USD") -> dict:
    return {"facts": {"us-gaap": {tag: {"units": {unit: entries}}}}}


class TestFiscalYearDerivation:
    def test_year_comes_from_the_period_not_the_report(self):
        """The whole bug in one assertion.

        `fy` is 2025 because the fact appeared in the FY2025 10-K. The period it
        describes ended in 2023.
        """
        assert fiscal_year_of(date(2023, 12, 31), reported_fy=2025) == 2023

    def test_january_year_end_belongs_to_that_calendar_year(self):
        """NVIDIA's fiscal 2024 ended 2024-01-28 and NVIDIA calls it fiscal 2024."""
        assert fiscal_year_of(date(2024, 1, 28), reported_fy=2024) == 2024

    def test_falls_back_to_the_reported_year_without_a_period(self):
        assert fiscal_year_of(None, reported_fy=2024) == 2024


class TestFullYearDetection:
    def test_a_standard_year(self):
        assert covers_full_year(date(2024, 1, 1), date(2024, 12, 31))

    def test_a_52_53_week_year(self):
        """Apple and Costco end on a weekday, so the span is never exactly 365."""
        assert covers_full_year(date(2023, 10, 1), date(2024, 9, 28))

    def test_a_quarter_is_not_a_year(self):
        assert not covers_full_year(date(2024, 1, 1), date(2024, 3, 31))

    def test_nine_months_is_not_a_year(self):
        """The stub that a 10-K carries alongside the annual figure."""
        assert not covers_full_year(date(2024, 1, 1), date(2024, 9, 30))

    def test_instantaneous_facts_have_no_duration(self):
        assert not covers_full_year(None, date(2024, 12, 31))


class TestComparativesAreFiledUnderTheirOwnYear:
    def test_three_comparatives_become_three_years(self):
        """The exact failure: one report, three years, all stamped fy=2025.

        Before the fix these collapsed into fiscal year 2025 and the golden set
        got `num-CAT-Revenues-2023` and `num-CAT-Revenues-2024` with identical
        expected values.
        """
        facts = extract_facts(
            "0000018230",
            _payload(
                [
                    {"fy": 2025, "fp": "FY", "form": "10-K", "val": 64_809_000_000,
                     "start": "2025-01-01", "end": "2025-12-31", "accn": "0000018230-26-000010"},
                    {"fy": 2025, "fp": "FY", "form": "10-K", "val": 67_060_000_000,
                     "start": "2024-01-01", "end": "2024-12-31", "accn": "0000018230-26-000010"},
                    {"fy": 2025, "fp": "FY", "form": "10-K", "val": 66_984_000_000,
                     "start": "2023-01-01", "end": "2023-12-31", "accn": "0000018230-26-000010"},
                ]
            ),
        )
        by_year = {f.fiscal_year: f.value for f in facts}
        assert by_year == {
            2025: 64_809_000_000,
            2024: 67_060_000_000,
            2023: 66_984_000_000,
        }

    def test_no_year_ends_up_with_two_values(self):
        facts = extract_facts(
            "0000018230",
            _payload(
                [
                    {"fy": 2025, "fp": "FY", "form": "10-K", "val": 2_148_000_000,
                     "start": "2025-01-01", "end": "2025-12-31", "accn": "0000018230-26-000010"},
                    {"fy": 2025, "fp": "FY", "form": "10-K", "val": 2_108_000_000,
                     "start": "2024-01-01", "end": "2024-12-31", "accn": "0000018230-26-000010"},
                ],
                tag="ResearchAndDevelopmentExpense",
            ),
        )
        assert conflicting_years(facts) == []

    def test_original_filing_wins_over_a_later_restatement(self):
        """Two filings report FY2024; the FY2024 10-K is the one in the corpus.

        Grading against a figure restated in a later filing would mark correct
        retrieval from the FY2024 text as wrong.
        """
        facts = extract_facts(
            "0000018230",
            _payload(
                [
                    {"fy": 2025, "fp": "FY", "form": "10-K", "val": 67_999_000_000,
                     "start": "2024-01-01", "end": "2024-12-31", "accn": "0000018230-26-000010"},
                    {"fy": 2024, "fp": "FY", "form": "10-K", "val": 67_060_000_000,
                     "start": "2024-01-01", "end": "2024-12-31", "accn": "0000018230-25-000008"},
                ]
            ),
        )
        assert len(facts) == 1
        assert facts[0].value == 67_060_000_000


class TestNonAnnualFactsAreExcluded:
    def test_a_quarterly_duration_is_dropped(self):
        facts = extract_facts(
            "0000018230",
            _payload(
                [
                    {"fy": 2025, "fp": "FY", "form": "10-K", "val": 64_809_000_000,
                     "start": "2025-01-01", "end": "2025-12-31", "accn": "a"},
                    {"fy": 2025, "fp": "FY", "form": "10-K", "val": 17_000_000_000,
                     "start": "2025-10-01", "end": "2025-12-31", "accn": "a"},
                ]
            ),
        )
        assert [f.value for f in facts] == [64_809_000_000]

    def test_a_balance_sheet_fact_at_a_year_end_is_kept(self):
        """Instantaneous facts have no start, so the year-end set decides."""
        payload = {
            "facts": {
                "us-gaap": {
                    "Revenues": {"units": {"USD": [
                        {"fy": 2025, "fp": "FY", "form": "10-K", "val": 64_809_000_000,
                         "start": "2025-01-01", "end": "2025-12-31", "accn": "a"},
                    ]}},
                    "Assets": {"units": {"USD": [
                        {"fy": 2025, "fp": "FY", "form": "10-K", "val": 87_476_000_000,
                         "end": "2025-12-31", "accn": "a"},
                    ]}},
                }
            }
        }
        facts = extract_facts("0000018230", payload)
        assets = [f for f in facts if f.tag == "Assets"]
        assert len(assets) == 1
        assert assets[0].fiscal_year == 2025

    def test_a_balance_sheet_fact_at_a_quarter_end_is_dropped(self):
        """A 10-K carries interim balances; those are not annual facts."""
        payload = {
            "facts": {
                "us-gaap": {
                    "Revenues": {"units": {"USD": [
                        {"fy": 2025, "fp": "FY", "form": "10-K", "val": 64_809_000_000,
                         "start": "2025-01-01", "end": "2025-12-31", "accn": "a"},
                    ]}},
                    "Assets": {"units": {"USD": [
                        {"fy": 2025, "fp": "FY", "form": "10-K", "val": 87_476_000_000,
                         "end": "2025-12-31", "accn": "a"},
                        {"fy": 2025, "fp": "FY", "form": "10-K", "val": 85_000_000_000,
                         "end": "2025-06-30", "accn": "a"},
                    ]}},
                }
            }
        }
        facts = extract_facts("0000018230", payload)
        assert [f.value for f in facts if f.tag == "Assets"] == [87_476_000_000]

    def test_non_10k_forms_are_filtered(self):
        facts = extract_facts(
            "0000018230",
            _payload(
                [
                    {"fy": 2025, "fp": "Q2", "form": "10-Q", "val": 1,
                     "start": "2025-01-01", "end": "2025-12-31", "accn": "a"},
                ]
            ),
        )
        assert facts == []


class TestConflictDetector:
    def test_reports_a_group_with_two_values(self):
        from edgar_intel.ingest.xbrl import Fact

        def f(year, value, accn):
            return Fact("c", "us-gaap", "Revenues", "USD", year, "FY",
                        None, date(year, 12, 31), value, accn, "10-K")

        assert conflicting_years([f(2024, 1.0, "a"), f(2024, 2.0, "b")]) == [
            ("c", "Revenues", "USD", 2024)
        ]

    def test_silent_when_clean(self):
        from edgar_intel.ingest.xbrl import Fact

        facts = [
            Fact("c", "us-gaap", "Revenues", "USD", y, "FY", None,
                 date(y, 12, 31), float(y), "a", "10-K")
            for y in (2023, 2024, 2025)
        ]
        assert conflicting_years(facts) == []
