"""The fact tools tell the model which tags exist.

The catalogue rendered tool names and argument types, never field descriptions,
so the model recalled XBRL names from memory. In the golden-set run with the
citation fix, 55 of 217 fact calls named a tag no fact carries,
`CashAndCashEquivalents` and `epsDiluted` most often. `get_fact`'s miss lists the
tags the company reports; `compare_fact`'s said only "Missing", and comparisons
were the agent's weakest kind (30/48 against 42/48 for retrieve-then-answer).
"""

from __future__ import annotations

import pytest

from edgar_intel.agent import tools
from edgar_intel.ingest.xbrl import CORE_TAGS


class NoRows:
    """A database holding nothing under the tag asked for."""

    def query(self, sql, params=None):
        return []

    def query_one(self, sql, params=None):
        return None


@pytest.fixture
def empty_db(monkeypatch):
    fake = NoRows()
    monkeypatch.setattr(tools.db, "query", fake.query)
    monkeypatch.setattr(tools.db, "query_one", fake.query_one)


def compare(tag: str) -> tools.ToolResult:
    return tools.call_tool(
        "compare_fact", {"ticker": "AAPL", "tag": tag, "year_a": 2023, "year_b": 2024}
    )


def test_the_catalogue_names_every_tag_the_fact_tools_hold():
    catalogue = tools.tool_catalogue()
    for tag in CORE_TAGS:
        assert tag in catalogue
    assert "'revenue'" in catalogue


@pytest.mark.parametrize("guess", ["CashAndCashEquivalents", "epsDiluted", "EPS_DILUTED"])
def test_a_guessed_tag_is_told_which_tags_exist(empty_db, guess):
    result = compare(guess)
    assert not result.ok
    assert f"Missing {guess} for AAPL" in result.summary
    assert "No fact carries that tag" in result.summary
    assert "CashAndCashEquivalentsAtCarryingValue" in result.summary
    assert "EarningsPerShareDiluted" in result.summary


@pytest.mark.parametrize("tag", ["Assets", "revenue"])
def test_a_real_tag_with_missing_years_gets_no_vocabulary_lecture(empty_db, tag):
    result = compare(tag)
    assert not result.ok
    assert "No fact carries that tag" not in result.summary
