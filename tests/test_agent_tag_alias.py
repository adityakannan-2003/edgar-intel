"""`revenue` is a word, not an XBRL tag.

The agent passes the user's word to `get_fact` as the tag. No XBRL fact is
tagged "revenue": the figure is filed as
`RevenueFromContractWithCustomerExcludingAssessedTax` (or, for CAT, NVDA, PG and
UNH, as `Revenues`). So the lookup matched nothing, the tool said "No revenue
reported", and the agent escalated a question whose answer was in the database.

The alias is deterministic: one word maps to a fixed, ordered list of tags, and
the first tag the company reported for that year wins. These tests pin the case
that escalated, the fallback, and two things an alias must never do: rewrite a
real tag, or disagree with the tag the golden set grades against.

`compare_fact` and `compute_ratio` take the same word and resolve it the same
way, with one difference: a comparison uses one tag for both years, because a
change between two different concepts is not a change.
"""

from __future__ import annotations

import json

import pytest

from edgar_intel.agent import loop as agent_loop
from edgar_intel.agent import tools
from edgar_intel.providers.base import Completion

CONTRACT = "RevenueFromContractWithCustomerExcludingAssessedTax"

# Values from the golden set, accessions from the ingested 10-Ks, so the fake
# table holds what the real one does.
JNJ_FY2025 = {
    "value": 94_193_000_000.0,
    "unit": "USD",
    "tag": CONTRACT,
    "fiscal_year": 2025,
    "accession": "0000200406-26-000016",
    "name": "JOHNSON & JOHNSON",
}
CAT_FY2025 = {
    "value": 67_589_000_000.0,
    "unit": "USD",
    "tag": "Revenues",
    "fiscal_year": 2025,
    "accession": "0000018230-26-000008",
    "name": "CATERPILLAR INC",
}
JNJ_FY2024 = {
    "value": 88_821_000_000.0,
    "unit": "USD",
    "tag": CONTRACT,
    "fiscal_year": 2024,
    "accession": "0000200406-25-000038",
    "name": "JOHNSON & JOHNSON",
}
JNJ_NET_INCOME_FY2025 = {
    **JNJ_FY2025,
    "value": 26_804_000_000.0,
    "tag": "NetIncomeLoss",
}
CAT_FY2023 = {
    **CAT_FY2025,
    "value": 67_060_000_000.0,
    "fiscal_year": 2023,
    "accession": "0000018230-24-000009",
}
CAT_FY2024 = {
    **CAT_FY2025,
    "value": 64_809_000_000.0,
    "fiscal_year": 2024,
    "accession": "0000018230-25-000008",
}
CAT_OPERATING_INCOME_FY2025 = {
    **CAT_FY2025,
    "value": 11_151_000_000.0,
    "tag": "OperatingIncomeLoss",
}
# NVIDIA switched tags: the contract tag through FY2022, `Revenues` from FY2023.
NVDA_CONTRACT_FY2022 = {
    "value": 26_914_000_000.0,
    "unit": "USD",
    "tag": CONTRACT,
    "fiscal_year": 2022,
    "accession": "0001045810-22-000036",
    "name": "NVIDIA CORP",
}
NVDA_REVENUES_FY2022 = {**NVDA_CONTRACT_FY2022, "tag": "Revenues"}
NVDA_REVENUES_FY2023 = {
    **NVDA_REVENUES_FY2022,
    "value": 26_974_000_000.0,
    "fiscal_year": 2023,
    "accession": "0001045810-23-000017",
}


class FakeFacts:
    """`xbrl_facts` joined to `companies`, answering the queries the fact tools make."""

    def __init__(self, *rows: tuple[str, dict]) -> None:
        self.rows = {(ticker, r["tag"], r["fiscal_year"]): r for ticker, r in rows}
        self.tags_queried: list[str] = []

    def query_one(self, sql, params):
        ticker, tag, fiscal_year, fiscal_period = params
        self.tags_queried.append(tag)
        if fiscal_period != "FY":
            return None
        return self.rows.get((ticker, tag, fiscal_year))

    def query(self, sql, params=None):
        if "x.fiscal_year = ANY" in sql:  # compare_fact: one tag, two years
            ticker, tag, years = params
            self.tags_queried.append(tag)
            return [
                {
                    "fiscal_year": y,
                    "value": r["value"],
                    "unit": r["unit"],
                    "prior_year_value": r.get("prior_year_value"),
                }
                for (t, g, y), r in self.rows.items()
                if t == ticker and g == tag and y in years
            ]
        if "x.tag = ANY" in sql:  # compute_ratio: several tags, one year
            ticker, tags, fiscal_year = params
            self.tags_queried.extend(tags)
            return [
                {"tag": g, "value": r["value"], "unit": r["unit"]}
                for (t, g, y), r in self.rows.items()
                if t == ticker and g in tags and y == fiscal_year
            ]
        ticker, fiscal_year = params  # get_fact's list of available tags
        return [{"tag": tag} for (t, tag, y) in self.rows if t == ticker and y == fiscal_year]


@pytest.fixture
def facts(monkeypatch):
    def _install(*rows: tuple[str, dict]) -> FakeFacts:
        fake = FakeFacts(*rows)
        monkeypatch.setattr(tools.db, "query_one", fake.query_one)
        monkeypatch.setattr(tools.db, "query", fake.query)
        return fake

    return _install


def get_fact(ticker: str, tag: str, fiscal_year: int) -> tools.ToolResult:
    """Through `call_tool`, the path the agent loop takes."""
    return tools.call_tool("get_fact", {"ticker": ticker, "tag": tag, "fiscal_year": fiscal_year})


def compare_fact(ticker: str, tag: str, year_a: int, year_b: int) -> tools.ToolResult:
    return tools.call_tool(
        "compare_fact", {"ticker": ticker, "tag": tag, "year_a": year_a, "year_b": year_b}
    )


def compute_ratio(ticker: str, numerator: str, denominator: str, year: int) -> tools.ToolResult:
    return tools.call_tool(
        "compute_ratio",
        {
            "ticker": ticker,
            "numerator_tag": numerator,
            "denominator_tag": denominator,
            "fiscal_year": year,
        },
    )


class ModelThatSaysRevenue:
    """Asks `get_fact` for "revenue", then does what the real model did.

    With a figure in the tool output it answers from it. With "No revenue
    reported" it escalates, which is the failure this file exists for.
    """

    name = "scripted"

    def complete(
        self, prompt, *, system=None, model=None, temperature=0.0, max_tokens=1024, json_schema=None
    ) -> Completion:
        if "TOOL get_fact" not in prompt:
            reply = {
                "thought": "look it up",
                "tool": "get_fact",
                "args": {"ticker": "JNJ", "tag": "revenue", "fiscal_year": 2025},
            }
        elif "-> ok" in prompt:
            reply = {
                "thought": "found it",
                "answer": "Johnson & Johnson's total revenue for FY2025 was $94.19 billion.",
                "citations": ["xbrl:0000200406-26-000016"],
                "confidence": 0.9,
            }
        else:
            reply = {
                "thought": "no figure",
                "tool": "escalate_to_human",
                "args": {"reason": "get_fact found no revenue for JNJ FY2025"},
            }
        return Completion(
            text=json.dumps(reply), prompt_tokens=10, completion_tokens=5, model="scripted"
        )


class TestTheCaseThatEscalated:
    def test_revenue_returns_the_reported_figure(self, facts):
        """The failing call as the agent made it: the user's word, not the tag."""
        facts(("JNJ", JNJ_FY2025))
        result = get_fact("JNJ", "revenue", 2025)
        assert result.ok, result.summary
        assert result.data["tag"] == CONTRACT
        assert result.data["value"] == 94_193_000_000.0
        assert "$94.19 billion" in result.summary
        assert result.citations == ["xbrl:0000200406-26-000016"]

    def test_the_agent_answers_instead_of_escalating(self, facts, monkeypatch):
        facts(("JNJ", JNJ_FY2025))
        monkeypatch.setattr(agent_loop, "get_llm", ModelThatSaysRevenue)
        run = agent_loop.run_agent(
            "What was Johnson & Johnson's total revenue in fiscal 2025?", persist=False
        )
        assert run.outcome == "answered", run.render()
        assert "$94.19 billion" in run.answer
        assert run.citations == ["xbrl:0000200406-26-000016"]


class TestTheAlias:
    @pytest.mark.parametrize("word", ["revenue", "Revenue", "REVENUE", "  revenue "])
    def test_case_and_surrounding_whitespace_do_not_matter(self, facts, word):
        facts(("JNJ", JNJ_FY2025))
        result = get_fact("JNJ", word, 2025)
        assert result.ok, result.summary
        assert result.data["tag"] == CONTRACT

    def test_falls_back_to_revenues_where_that_is_all_the_filer_reports(self, facts):
        """CAT, NVDA, PG and UNH: 13 of the golden set's 20 revenue cases."""
        fake = facts(("CAT", CAT_FY2025))
        result = get_fact("CAT", "revenue", 2025)
        assert result.ok, result.summary
        assert result.data["tag"] == "Revenues"
        assert result.data["value"] == 67_589_000_000.0
        assert fake.tags_queried == [CONTRACT, "Revenues"]

    def test_the_contract_tag_wins_when_both_are_reported(self, facts):
        """The order decides, not whichever row the database returns first."""
        # Constructed: a second revenue tag for the same year, deliberately different.
        also_revenues = {**JNJ_FY2025, "tag": "Revenues", "value": 95_000_000_000.0}
        fake = facts(("JNJ", also_revenues), ("JNJ", JNJ_FY2025))
        result = get_fact("JNJ", "revenue", 2025)
        assert result.data["tag"] == CONTRACT
        assert result.data["value"] == 94_193_000_000.0
        assert fake.tags_queried == [CONTRACT]

    def test_a_real_tag_is_never_rewritten(self, facts):
        """`Revenues` means `Revenues`, even where the contract tag exists too."""
        also_revenues = {**JNJ_FY2025, "tag": "Revenues", "value": 95_000_000_000.0}
        fake = facts(("JNJ", also_revenues), ("JNJ", JNJ_FY2025))
        result = get_fact("JNJ", "Revenues", 2025)
        assert result.data["tag"] == "Revenues"
        assert result.data["value"] == 95_000_000_000.0
        assert fake.tags_queried == ["Revenues"]

    def test_other_tags_are_queried_as_given(self, facts):
        fake = facts()
        get_fact("JNJ", "NetIncomeLoss", 2025)
        assert fake.tags_queried == ["NetIncomeLoss"]

    def test_a_year_with_no_revenue_still_fails_and_names_what_it_tried(self, facts):
        """The alias must not turn a real gap into an answer."""
        facts(("JNJ", JNJ_FY2025))
        result = get_fact("JNJ", "revenue", 2031)
        assert not result.ok
        assert result.summary.startswith("No revenue reported for JNJ FY2031")
        assert f"tried {CONTRACT}, Revenues" in result.summary


class TestCompareFact:
    def test_revenue_is_compared_under_the_contract_tag(self, facts):
        facts(("JNJ", JNJ_FY2024), ("JNJ", JNJ_FY2025))
        result = compare_fact("JNJ", "revenue", 2024, 2025)
        assert result.ok, result.summary
        assert result.data["tag"] == CONTRACT
        assert result.data["year_a"] == 88_821_000_000.0
        assert result.data["year_b"] == 94_193_000_000.0
        assert result.data["pct_change"] == 6.0481
        assert "increased 6.0%" in result.summary

    def test_falls_back_to_revenues_and_matches_the_golden_set(self, facts):
        """`yoy-CAT-Revenues-2023-2024` expects -3.3567."""
        fake = facts(("CAT", CAT_FY2023), ("CAT", CAT_FY2024))
        result = compare_fact("CAT", "revenue", 2023, 2024)
        assert result.ok, result.summary
        assert result.data["tag"] == "Revenues"
        assert result.data["pct_change"] == -3.3567
        assert fake.tags_queried == [CONTRACT, "Revenues"]

    def test_a_filer_that_switched_tags_is_compared_within_one(self, facts):
        """NVIDIA's contract tag stops at FY2022, so it cannot cover FY2022 to
        FY2023. `Revenues` covers both, and both figures come from it."""
        fake = facts(
            ("NVDA", NVDA_CONTRACT_FY2022),
            ("NVDA", NVDA_REVENUES_FY2022),
            ("NVDA", NVDA_REVENUES_FY2023),
        )
        result = compare_fact("NVDA", "revenue", 2022, 2023)
        assert result.ok, result.summary
        assert result.data["tag"] == "Revenues"
        assert result.data["pct_change"] == 0.2229
        assert fake.tags_queried == [CONTRACT, "Revenues"]

    def test_the_prior_year_comes_from_the_tag_that_was_chosen(self, facts):
        """The alias picks the tag; the D12 basis must then come from that
        tag's own later-year filing, not from the tag tried first. Constructed:
        FY2023's filing restates FY2022 as 26,000M under `Revenues`."""
        restated = {**NVDA_REVENUES_FY2023, "prior_year_value": 26_000_000_000.0}
        facts(("NVDA", NVDA_CONTRACT_FY2022), ("NVDA", NVDA_REVENUES_FY2022), ("NVDA", restated))
        result = compare_fact("NVDA", "revenue", 2022, 2023)
        assert result.ok, result.summary
        assert result.data["tag"] == "Revenues"
        assert result.data["year_a"] == 26_000_000_000.0
        assert result.data["pct_change"] == 3.7462

    def test_two_different_tags_are_not_a_change(self, facts):
        """Constructed: no one tag covers both years. Mixing them would report
        the difference between two concepts as growth."""
        facts(("JNJ", JNJ_FY2024), ("JNJ", {**JNJ_FY2025, "tag": "Revenues"}))
        result = compare_fact("JNJ", "revenue", 2024, 2025)
        assert not result.ok
        assert "reported under different tags in FY2024 and FY2025" in result.summary
        assert result.data["have"] == [2024, 2025]

    def test_a_missing_year_still_fails_and_names_what_it_tried(self, facts):
        facts(("JNJ", JNJ_FY2024), ("JNJ", JNJ_FY2025))
        result = compare_fact("JNJ", "revenue", 2025, 2031)
        assert not result.ok
        assert result.summary.startswith("Missing revenue for JNJ in [2031]")
        assert f"tried {CONTRACT}, Revenues" in result.summary

    def test_a_real_tag_is_queried_as_given(self, facts):
        fake = facts(("JNJ", JNJ_NET_INCOME_FY2025))
        result = compare_fact("JNJ", "NetIncomeLoss", 2024, 2025)
        assert not result.ok
        assert result.summary == "Missing NetIncomeLoss for JNJ in [2024]."
        assert fake.tags_queried == ["NetIncomeLoss"]


class TestComputeRatio:
    def test_revenue_resolves_as_the_denominator(self, facts):
        """Net margin, the ratio a model reaches for with the word "revenue"."""
        facts(("JNJ", JNJ_NET_INCOME_FY2025), ("JNJ", JNJ_FY2025))
        result = compute_ratio("JNJ", "NetIncomeLoss", "revenue", 2025)
        assert result.ok, result.summary
        assert result.data["ratio"] == 0.284565
        assert f"NetIncomeLoss / {CONTRACT}" in result.summary

    def test_falls_back_to_revenues(self, facts):
        facts(("CAT", CAT_OPERATING_INCOME_FY2025), ("CAT", CAT_FY2025))
        result = compute_ratio("CAT", "OperatingIncomeLoss", "revenue", 2025)
        assert result.ok, result.summary
        assert result.data["ratio"] == 0.164982
        assert "OperatingIncomeLoss / Revenues" in result.summary

    def test_the_contract_tag_wins_whichever_row_comes_back_first(self, facts):
        also_revenues = {**JNJ_FY2025, "tag": "Revenues", "value": 95_000_000_000.0}
        facts(("JNJ", also_revenues), ("JNJ", JNJ_NET_INCOME_FY2025), ("JNJ", JNJ_FY2025))
        result = compute_ratio("JNJ", "NetIncomeLoss", "revenue", 2025)
        assert result.data["ratio"] == 0.284565
        assert result.data[CONTRACT] == 94_193_000_000.0
        assert "Revenues" not in result.data

    def test_a_missing_side_fails_and_names_what_it_tried(self, facts):
        facts(("JNJ", JNJ_NET_INCOME_FY2025))
        result = compute_ratio("JNJ", "NetIncomeLoss", "revenue", 2025)
        assert not result.ok
        assert result.summary == (
            f"Need both NetIncomeLoss and revenue (tried {CONTRACT}, Revenues); "
            "have ['NetIncomeLoss']."
        )


class TestAgreesWithTheGoldenSet:
    def test_resolution_order_is_the_golden_set_preference(self):
        """The golden set grades revenue against the first of these tags the
        company reports. Resolving in another order would answer with one tag's
        figure and be graded against the other's."""
        from edgar_intel.evals.goldenset import PREFERRED_TAG_FAMILIES

        assert list(tools.resolve_tag("revenue")) == PREFERRED_TAG_FAMILIES["revenue"]
        assert tools.resolve_tag("revenue")[0] == CONTRACT
