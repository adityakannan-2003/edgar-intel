"""Evidence labels come only from the filing search can reach (D15).

`search` filters chunks on the case's fiscal year. The evidence linker accepted
that year *or any later one*, because a later 10-K reprints earlier years as
comparatives. Those chunks contain the figure, but search can never return
them. On the rebuilt golden set 178 of 919 label references (19%) were out of
reach, and recall@5 read 0.320 where the reachable labels gave 0.408.

These tests record the SQL the linker sends, with no database, and pin both
sides of the contract: every linker query filters on exactly the case's year,
and search's own filter is the same exact match.
"""

from __future__ import annotations

import pytest

from edgar_intel.evals.schemas import EvalCase


@pytest.fixture
def queries(monkeypatch) -> list[tuple[str, tuple]]:
    import edgar_intel.db as db

    seen: list[tuple[str, tuple]] = []

    def query(sql, params=None):
        seen.append((sql, tuple(params or ())))
        return []

    monkeypatch.setattr(db, "query", query)
    return seen


def link(case: EvalCase) -> None:
    from edgar_intel.evals.goldenset import link_evidence

    link_evidence([case], "section_aware")


def single_hop() -> EvalCase:
    return EvalCase(
        case_id="num-JNJ-GrossProfit-2023", kind="numeric", question="q?",
        expected="$58.61 billion", expected_value=58_606_000_000.0, unit="USD",
        ticker="JNJ", fiscal_year=2023, tag="GrossProfit",
    )


def comparative() -> EvalCase:
    return EvalCase(
        case_id="yoy-JNJ-Assets-2023-2024", kind="numeric", question="q?",
        expected="increased 7.5%", expected_value=7.4882, unit="percent",
        ticker="JNJ", fiscal_year=2024, tag="Assets", difficulty="comparative",
        notes="FY2023=167,558,000,000, FY2024=180,104,000,000",
    )


def year_filters(seen) -> list[tuple[str, int]]:
    """(operator, year) for the fiscal-year clause of every query sent."""
    out = []
    for sql, params in seen:
        assert "f.fiscal_year >=" not in sql, sql
        assert "AND f.fiscal_year = %s" in sql, sql
        # Parameter order: strategy, ticker, fiscal_year, ...
        out.append(("=", params[2]))
    return out


class TestTheLinkerStaysInTheCasesFiling:
    def test_single_hop(self, queries):
        link(single_hop())
        assert queries, "the linker sent no query"
        assert set(year_filters(queries)) == {("=", 2023)}

    def test_comparative_uses_the_later_filing_for_both_values(self, queries):
        """The 2024 10-K prints 2023 beside 2024, and it is the filing search
        reaches for a question whose case year is 2024."""
        link(comparative())
        assert queries
        assert set(year_filters(queries)) == {("=", 2024)}

    def test_the_fallback_searches_the_same_filing(self, queries):
        """No chunk holds both values here, so it falls back to one value at a
        time, and still only in the 2024 filing, including the 2023 value."""
        link(comparative())
        both = [q for q in queries if q[0].count("c.body ILIKE") == 2]
        one = [q for q in queries if q[0].count("c.body ILIKE") == 1]
        assert both and one
        assert {params[2] for _, params in one} == {2024}


class TestSearchFiltersTheSameWay:
    def test_search_filters_on_exactly_the_fiscal_year(self):
        from edgar_intel.retrieval.search import _filters

        clauses, params = _filters("JNJ", 2023, None)
        sql = " AND ".join(clauses) if isinstance(clauses, list) else str(clauses)
        assert "f.fiscal_year = %s" in sql
        assert 2023 in params
