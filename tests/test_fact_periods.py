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

import pytest

from edgar_intel.ingest.xbrl import (
    Fact,
    conflicting_years,
    covers_full_year,
    extract_facts,
    fiscal_year_ending,
    fiscal_year_of,
    misdated_years,
    missing_years,
    pick_research_and_development,
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

    @pytest.mark.parametrize(
        ("period_end", "fiscal_year"),
        [
            (date(2023, 1, 1), 2022),  # 10-K 0000200406-23-000016, fy 2022
            (date(2022, 1, 2), 2021),
            (date(2021, 1, 3), 2020),
            (date(2017, 1, 1), 2016),
        ],
    )
    def test_early_january_year_end_belongs_to_the_year_before(self, period_end, fiscal_year):
        """JNJ's year ends on the Sunday nearest December 31, sometimes in January.

        Its FY2022 10-K covers 2022-01-03 -> 2023-01-01 and says fiscal 2022.
        The year of `end` filed it under 2023 (D13).
        """
        assert fiscal_year_of(period_end, reported_fy=fiscal_year) == fiscal_year

    @pytest.mark.parametrize(
        ("period_end", "fiscal_year"),
        [
            (date(2023, 12, 31), 2023),  # JNJ, the same rule ending in December
            (date(2024, 12, 29), 2024),  # JNJ
            (date(2015, 1, 25), 2015),  # NVDA, the earliest its year has ended
            (date(2021, 1, 31), 2021),  # NVDA, a 53-week year
            (date(2024, 9, 28), 2024),  # AAPL
            (date(2023, 9, 3), 2023),  # COST, a 53-week year
        ],
    )
    def test_other_year_ends_keep_the_year_they_end_in(self, period_end, fiscal_year):
        assert fiscal_year_of(period_end, reported_fy=0) == fiscal_year

    def test_filings_use_the_same_rule_as_facts(self):
        """`filings.fiscal_year` and `xbrl_facts.fiscal_year` must agree, or the
        golden set drops a covered year and verify-facts reports false drift."""
        for end in (date(2023, 1, 1), date(2023, 12, 31), date(2024, 1, 28)):
            assert fiscal_year_ending(end) == fiscal_year_of(end, reported_fy=0)


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

def test_preferred_r_and_d_tag_wins():
    facts = [
        Fact(
            cik="0000200406",
            taxonomy="us-gaap",
            tag="ResearchAndDevelopmentExpense",
            unit="USD",
            fiscal_year=2025,
            fiscal_period="FY",
            period_start=None,
            period_end=None,
            value=109_000_000,
            accession=None,
            form="10-K",
            ),
        Fact(
            cik="0000200406",
            taxonomy="us-gaap",
            tag="ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
            unit="USD",
            fiscal_year=2025,
            fiscal_period="FY",
            period_start=None,
            period_end=None,
            value=17_000_000_000,
            accession=None,
            form="10-K",
        ),
    ]

    picked = pick_research_and_development(facts, 2025)

    assert picked is not None
    assert picked.tag == "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"
    assert picked.value == 17_000_000_000

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


JNJ = "0000200406"
JNJ_FY2022_10K = "0000200406-23-000016"  # fy 2022, year ended 2023-01-01
JNJ_FY2023_10K = "0000200406-24-000013"  # fy 2023, year ended 2023-12-31


class TestEarlyJanuaryYearEnds:
    """JNJ's FY2022 and FY2023 10-Ks, as `companyfacts` reports them."""

    def test_fiscal_2022_does_not_displace_fiscal_2023(self):
        """The collision that dropped JNJ's real fiscal 2023.

        With the year of `end`, 2022-01-03 -> 2023-01-01 became "2023". The
        FY2022 10-K is the earlier accession, so it won the key, and
        `xbrl_facts` held $94.94 billion as JNJ's 2023 revenue instead of $85.16
        billion. Fiscal 2020 was filed as 2021, and nothing as 2020.
        """
        facts = extract_facts(
            JNJ,
            _payload(
                [
                    {"fy": 2022, "fp": "FY", "form": "10-K", "val": 82_584_000_000,
                     "start": "2019-12-30", "end": "2021-01-03", "accn": JNJ_FY2022_10K},
                    {"fy": 2022, "fp": "FY", "form": "10-K", "val": 93_775_000_000,
                     "start": "2021-01-04", "end": "2022-01-02", "accn": JNJ_FY2022_10K},
                    {"fy": 2022, "fp": "FY", "form": "10-K", "val": 94_943_000_000,
                     "start": "2022-01-03", "end": "2023-01-01", "accn": JNJ_FY2022_10K},
                    {"fy": 2023, "fp": "FY", "form": "10-K", "val": 79_990_000_000,
                     "start": "2022-01-03", "end": "2023-01-01", "accn": JNJ_FY2023_10K},
                    {"fy": 2023, "fp": "FY", "form": "10-K", "val": 85_159_000_000,
                     "start": "2023-01-02", "end": "2023-12-31", "accn": JNJ_FY2023_10K},
                ],
                tag="RevenueFromContractWithCustomerExcludingAssessedTax",
            ),
        )
        by_year = {f.fiscal_year: f.value for f in facts}
        assert by_year == {
            2020: 82_584_000_000,
            2021: 93_775_000_000,
            2022: 94_943_000_000,  # as first reported, not the post-Kenvue 79.99bn
            2023: 85_159_000_000,
        }
        assert conflicting_years(facts) == []

    def test_a_balance_sheet_dated_in_early_january(self):
        """Instant facts have no start; the date alone must give the year."""
        payload = {
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                        {"fy": 2022, "fp": "FY", "form": "10-K", "val": 94_943_000_000,
                         "start": "2022-01-03", "end": "2023-01-01", "accn": JNJ_FY2022_10K},
                        {"fy": 2023, "fp": "FY", "form": "10-K", "val": 85_159_000_000,
                         "start": "2023-01-02", "end": "2023-12-31", "accn": JNJ_FY2023_10K},
                    ]}},
                    "Assets": {"units": {"USD": [
                        {"fy": 2022, "fp": "FY", "form": "10-K", "val": 187_378_000_000,
                         "end": "2023-01-01", "accn": JNJ_FY2022_10K},
                        {"fy": 2023, "fp": "FY", "form": "10-K", "val": 167_558_000_000,
                         "end": "2023-12-31", "accn": JNJ_FY2023_10K},
                    ]}},
                }
            }
        }
        assets = {f.fiscal_year: f.value for f in extract_facts(JNJ, payload) if f.tag == "Assets"}
        assert assets == {2022: 187_378_000_000, 2023: 167_558_000_000}


def _series(rows, cik=JNJ, tag="RevenueFromContractWithCustomerExcludingAssessedTax"):
    """Facts from (fiscal_year, period_start, period_end, value) rows."""
    return [
        Fact(cik, "us-gaap", tag, "USD", year, "FY", start, end, value, None, "10-K")
        for year, start, end, value in rows
    ]


# JNJ revenue exactly as ingested under the old rule (xbrl_facts, 29 Sep 2026).
JNJ_REVENUE_AS_INGESTED = [
    (2018, date(2018, 1, 1), date(2018, 12, 30), 81_581_000_000),
    (2019, date(2018, 12, 31), date(2019, 12, 29), 82_059_000_000),
    (2021, date(2019, 12, 30), date(2021, 1, 3), 82_584_000_000),
    (2022, date(2021, 1, 4), date(2022, 1, 2), 93_775_000_000),
    (2023, date(2022, 1, 3), date(2023, 1, 1), 94_943_000_000),
    (2024, date(2024, 1, 1), date(2024, 12, 29), 88_821_000_000),
    (2025, date(2024, 12, 30), date(2025, 12, 28), 94_193_000_000),
]


class TestMislabelledYearsAreDetected:
    """The checks `ingest verify-facts` runs without knowing the year rule.

    The drift check compares facts to the filing index, but the index derives
    its year the same way, and only covers the ingested filings. It could not
    have found JNJ's missing 2020. These can.
    """

    def test_jnj_as_ingested_is_flagged(self):
        facts = _series(JNJ_REVENUE_AS_INGESTED)
        tag = "RevenueFromContractWithCustomerExcludingAssessedTax"
        assert misdated_years(facts) == [
            # labelled two years apart, ended one year apart: 2020 is under "2021"
            (JNJ, tag, "USD", 2019, date(2019, 12, 29), 2021, date(2021, 1, 3)),
            # labelled adjacent, ended two years apart: real 2023 was displaced
            (JNJ, tag, "USD", 2023, date(2023, 1, 1), 2024, date(2024, 12, 29)),
        ]
        assert missing_years(facts) == [(JNJ, 2020)]

    def test_jnj_under_the_current_rule_is_clean(self):
        facts = _series(
            [
                (fiscal_year_of(end, reported_fy=0), start, end, value)
                for _, start, end, value in JNJ_REVENUE_AS_INGESTED
            ]
            + [(2023, date(2023, 1, 2), date(2023, 12, 31), 85_159_000_000)]
        )
        assert misdated_years(facts) == []
        assert missing_years(facts) == []

    def test_52_53_week_and_january_filers_are_clean(self):
        """A 53-week year or a late-January end must not read as a mislabel."""
        nvda = _series(
            [
                (2021, date(2020, 1, 27), date(2021, 1, 31), 16_675_000_000),
                (2022, date(2021, 2, 1), date(2022, 1, 30), 26_914_000_000),
                (2023, date(2022, 1, 31), date(2023, 1, 29), 26_974_000_000),
                (2024, date(2023, 1, 30), date(2024, 1, 28), 60_922_000_000),
            ],
            cik="0001045810",
            tag="Revenues",
        )
        cost = _series(
            [
                (2022, date(2021, 8, 30), date(2022, 8, 28), 1.0),
                (2023, date(2022, 8, 29), date(2023, 9, 3), 1.0),
                (2024, date(2023, 9, 4), date(2024, 9, 1), 1.0),
            ],
            cik="0000909832",
        )
        assert misdated_years(nvda + cost) == []
        assert missing_years(nvda + cost) == []

    def test_a_tag_may_skip_a_year_that_the_company_did_not(self):
        """Filers switch concepts. A gap in one tag, with the periods two years
        apart as labelled, is not a mislabel -- and the company still has a fact
        for the year under another tag."""
        rd = _series(
            [
                (2021, date(2021, 1, 4), date(2022, 1, 2), 1.0),
                (2023, date(2023, 1, 2), date(2023, 12, 31), 1.0),
            ],
            tag="ResearchAndDevelopmentExpense",
        )
        revenue = _series([(2022, date(2022, 1, 3), date(2023, 1, 1), 1.0)])
        assert misdated_years(rd + revenue) == []
        assert missing_years(rd + revenue) == []


NVDA = "0001045810"
NVDA_FY2023_10K = "0001045810-23-000017"
NVDA_FY2024_10K = "0001045810-24-000029"
NVDA_FY2025_10K = "0001045810-25-000023"  # first filing after the June 2024 10-for-1 split

# Diluted EPS exactly as `companyfacts` reports it in each 10-K.
NVDA_EPS = [
    {"fy": 2023, "fp": "FY", "form": "10-K", "val": 1.74,
     "start": "2022-01-31", "end": "2023-01-29", "accn": NVDA_FY2023_10K},
    {"fy": 2024, "fp": "FY", "form": "10-K", "val": 1.74,
     "start": "2022-01-31", "end": "2023-01-29", "accn": NVDA_FY2024_10K},
    {"fy": 2024, "fp": "FY", "form": "10-K", "val": 11.93,
     "start": "2023-01-30", "end": "2024-01-28", "accn": NVDA_FY2024_10K},
    {"fy": 2025, "fp": "FY", "form": "10-K", "val": 0.17,
     "start": "2022-01-31", "end": "2023-01-29", "accn": NVDA_FY2025_10K},
    {"fy": 2025, "fp": "FY", "form": "10-K", "val": 1.19,
     "start": "2023-01-30", "end": "2024-01-28", "accn": NVDA_FY2025_10K},
    {"fy": 2025, "fp": "FY", "form": "10-K", "val": 2.94,
     "start": "2024-01-29", "end": "2025-01-26", "accn": NVDA_FY2025_10K},
]


class TestComparisonsStayOnOneFilingsBasis:
    """D12: a year-over-year change must not straddle a stock split.

    Each year keeps its earliest filing's figure, so FY2024 EPS is the $11.93
    the FY2024 10-K reported. After the split, the FY2025 10-K printed FY2024
    as $1.19 beside FY2025's $2.94. Pairing $11.93 with $2.94 expected
    "decreased 75.4%" for a rise of 147%.
    """

    def test_each_fact_carries_its_own_filings_prior_year(self):
        facts = {
            f.fiscal_year: f
            for f in extract_facts(NVDA, _payload(NVDA_EPS, "EarningsPerShareDiluted", "USD/shares"))
        }
        # one year asked on its own: the figure as first reported
        assert {y: f.value for y, f in facts.items()} == {2023: 1.74, 2024: 11.93, 2025: 2.94}
        # the year before, as each of those filings printed it
        assert facts[2025].prior_year_value == 1.19
        assert facts[2024].prior_year_value == 1.74
        # the FY2023 10-K's FY2022 is not in the payload
        assert facts[2023].prior_year_value is None

    def test_balance_sheet_prior_year_comes_from_the_same_filing(self):
        payload = _payload(NVDA_EPS, "EarningsPerShareDiluted", "USD/shares")
        payload["facts"]["us-gaap"]["CommonStockSharesOutstanding"] = {"units": {"shares": [
            {"fy": 2024, "fp": "FY", "form": "10-K", "val": 2_464_000_000,
             "end": "2024-01-28", "accn": NVDA_FY2024_10K},
            {"fy": 2025, "fp": "FY", "form": "10-K", "val": 24_643_000_000,
             "end": "2024-01-28", "accn": NVDA_FY2025_10K},
            {"fy": 2025, "fp": "FY", "form": "10-K", "val": 24_477_000_000,
             "end": "2025-01-26", "accn": NVDA_FY2025_10K},
        ]}}
        shares = {
            f.fiscal_year: f
            for f in extract_facts(NVDA, payload)
            if f.tag == "CommonStockSharesOutstanding"
        }
        assert shares[2024].value == 2_464_000_000
        assert shares[2025].value == 24_477_000_000
        assert shares[2025].prior_year_value == 24_643_000_000

    def test_golden_yoy_case_uses_the_current_filings_comparative(self):
        import random

        from edgar_intel.evals.goldenset import _yoy_cases

        rows = [
            {"cik": NVDA, "tag": "EarningsPerShareDiluted", "unit": "USD/shares",
             "fiscal_year": 2024, "fiscal_period": "FY", "value": 11.93,
             "prior_year_value": 1.74},
            {"cik": NVDA, "tag": "EarningsPerShareDiluted", "unit": "USD/shares",
             "fiscal_year": 2025, "fiscal_period": "FY", "value": 2.94,
             "prior_year_value": 1.19},
        ]
        [case] = _yoy_cases(rows, "NVIDIA CORP", "NVDA", NVDA, random.Random(7))
        assert case.case_id == "yoy-NVDA-EarningsPerShareDiluted-2024-2025"
        assert case.expected == "increased 147.1%"
        assert case.expected_value == round((2.94 - 1.19) / 1.19 * 100, 4)
        assert case.notes.startswith("FY2024=1.19, FY2025=2.94")
        assert "originally reported as 11.93" in case.notes

    def test_golden_yoy_case_without_a_printed_prior_uses_that_years_value(self):
        """Rows ingested before the column existed, or a filing that printed no
        prior year: nothing to correct, so the old pairing stands."""
        import random

        from edgar_intel.evals.goldenset import _yoy_cases

        rows = [
            {"cik": "c", "tag": "Revenues", "unit": "USD", "fiscal_year": y,
             "fiscal_period": "FY", "value": v, "prior_year_value": None}
            for y, v in ((2024, 100.0), (2025, 110.0))
        ]
        [case] = _yoy_cases(rows, "X", "X", "c", random.Random(7))
        assert case.expected == "increased 10.0%"
        assert case.notes == "FY2024=100.00, FY2025=110.00"

    def test_agent_compare_fact_across_the_split(self, monkeypatch):
        from edgar_intel.agent import tools

        rows = [
            {"fiscal_year": 2024, "value": 11.93, "unit": "USD/shares", "prior_year_value": 1.74},
            {"fiscal_year": 2025, "value": 2.94, "unit": "USD/shares", "prior_year_value": 1.19},
        ]
        monkeypatch.setattr(tools.db, "query", lambda sql, params=None: rows)
        forward = tools.compare_fact(
            tools.CompareFactArgs(ticker="NVDA", tag="EarningsPerShareDiluted",
                                  year_a=2024, year_b=2025)
        )
        assert forward.data["year_a"] == 1.19
        assert forward.data["pct_change"] == round((2.94 - 1.19) / 1.19 * 100, 4)
        backward = tools.compare_fact(
            tools.CompareFactArgs(ticker="NVDA", tag="EarningsPerShareDiluted",
                                  year_a=2025, year_b=2024)
        )
        assert backward.data["year_b"] == 1.19


class TestFactsOnlyRefresh:
    """The rebuild path: corrected facts must replace stale ones, and nothing
    else may move. `upsert_facts` skips existing keys and `ingest run`
    deletes the chunk index through `chunks.section_id`."""

    @staticmethod
    def _fake_connection(monkeypatch, log):
        from contextlib import contextmanager

        from edgar_intel.ingest import pipeline

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params=None):
                log.append((" ".join(sql.split()), params))

            def executemany(self, sql, rows):
                log.append((" ".join(sql.split()), list(rows)))

        class Conn:
            def cursor(self):
                return Cursor()

        @contextmanager
        def connection():
            log.append("begin")
            yield Conn()
            log.append("commit")

        monkeypatch.setattr(pipeline.db, "connection", connection)

    def test_replace_deletes_then_inserts_in_one_transaction(self, monkeypatch):
        from edgar_intel.ingest.pipeline import replace_facts

        log = []
        self._fake_connection(monkeypatch, log)
        facts = extract_facts(NVDA, _payload(NVDA_EPS, "EarningsPerShareDiluted", "USD/shares"))
        assert replace_facts(NVDA, facts) == 3
        assert log[0] == "begin" and log[-1] == "commit"
        assert log[1] == ("DELETE FROM xbrl_facts WHERE cik = %s", (NVDA,))
        assert log[2][0].startswith("INSERT INTO xbrl_facts")
        inserted = log[2][1]
        assert {(r[4], r[8], r[11]) for r in inserted} == {
            (2023, 1.74, None), (2024, 11.93, 1.74), (2025, 2.94, 1.19),
        }

    def test_refresh_touches_only_facts(self, monkeypatch):
        from edgar_intel.ingest import pipeline

        log = []
        self._fake_connection(monkeypatch, log)
        monkeypatch.setattr(
            pipeline.db, "query",
            lambda sql, params=None: [{"cik": NVDA, "ticker": "NVDA"}, {"cik": JNJ, "ticker": "JNJ"}],
        )

        class Client:
            fetched: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def company_facts(self, cik):
                self.fetched.append(cik)
                return _payload(NVDA_EPS, "EarningsPerShareDiluted", "USD/shares")

        monkeypatch.setattr(pipeline, "EdgarClient", Client)
        stats = pipeline.refresh_facts(["nvda"])
        assert Client.fetched == [NVDA]
        assert (stats.companies, stats.facts, stats.filings, stats.sections) == (1, 3, 0, 0)
        assert [e for e in log if isinstance(e, tuple) and "DELETE" in e[0]] == [
            ("DELETE FROM xbrl_facts WHERE cik = %s", (NVDA,))
        ]
