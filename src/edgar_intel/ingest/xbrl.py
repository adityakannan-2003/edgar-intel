"""XBRL company facts -> verifiable numeric ground truth.

This is the load-bearing idea of the whole repo.

Most LLM evaluation projects grade generated text with another LLM. That
measures agreement, not correctness, and it degrades silently as models change.
Here, the SEC publishes the same numbers that appear in the filing text as
structured XBRL facts. So for any question of the form "what was NVIDIA's
FY2024 revenue", there is an authoritative answer that did not come from a
model, and the eval can check exact numeric agreement within a tolerance.

That gives the evaluation an external anchor. The LLM judge is then only used
for the genuinely open-ended narrative questions, and its trustworthiness is
itself measured (see evals/judge.py) rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

# The tags worth generating questions from: widely reported, unambiguous, and
# present for essentially every large filer.
CORE_TAGS: dict[str, str] = {
    "Revenues": "total revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "total revenue",
    "NetIncomeLoss": "net income",
    "OperatingIncomeLoss": "operating income",
    "ResearchAndDevelopmentExpense": "research and development expense",
    "Assets": "total assets",
    "Liabilities": "total liabilities",
    "StockholdersEquity": "total stockholders' equity",
    "CashAndCashEquivalentsAtCarryingValue": "cash and cash equivalents",
    "GrossProfit": "gross profit",
    "EarningsPerShareDiluted": "diluted earnings per share",
    "CommonStockSharesOutstanding": "common shares outstanding",
}


@dataclass(slots=True)
class Fact:
    cik: str
    taxonomy: str
    tag: str
    unit: str
    fiscal_year: int
    fiscal_period: str
    period_start: date | None
    period_end: date | None
    value: float
    accession: str | None
    form: str | None

    def label(self) -> str:
        return CORE_TAGS.get(self.tag, self.tag)


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def extract_facts(
    cik: str,
    company_facts: dict[str, Any],
    tags: dict[str, str] | None = None,
    forms: tuple[str, ...] = ("10-K",),
) -> list[Fact]:
    """Flatten companyfacts JSON into Fact rows for the tags we care about.

    Two subtleties that cost real debugging time if missed:

    1. The same (tag, period) is reported in multiple filings -- the original
       10-K and then again as a comparative in later ones. We keep the earliest
       accession for a given period so a "FY2022 revenue" question resolves to
       what was reported for FY2022, not to a restated figure.
    2. `frame`-less entries and instantaneous facts (balance-sheet items) have
       no `start`, only `end`. Both are kept; duration facts are distinguished
       by having a start date.
    """
    tags = tags or CORE_TAGS
    facts_root = company_facts.get("facts", {})
    out: dict[tuple, Fact] = {}

    for taxonomy in ("us-gaap", "dei"):
        tax_block = facts_root.get(taxonomy, {})
        for tag, tag_block in tax_block.items():
            if tag not in tags:
                continue
            for unit, entries in (tag_block.get("units") or {}).items():
                for entry in entries:
                    form = entry.get("form")
                    if forms and form not in forms:
                        continue
                    fy = entry.get("fy")
                    fp = entry.get("fp")
                    val = entry.get("val")
                    if fy is None or fp is None or val is None:
                        continue
                    period_end = _parse_date(entry.get("end"))
                    key = (taxonomy, tag, unit, int(fy), str(fp), period_end)
                    candidate = Fact(
                        cik=cik,
                        taxonomy=taxonomy,
                        tag=tag,
                        unit=unit,
                        fiscal_year=int(fy),
                        fiscal_period=str(fp),
                        period_start=_parse_date(entry.get("start")),
                        period_end=period_end,
                        value=float(val),
                        accession=entry.get("accn"),
                        form=form,
                    )
                    existing = out.get(key)
                    if existing is None or _is_earlier(candidate, existing):
                        out[key] = candidate
    return list(out.values())


def _is_earlier(a: Fact, b: Fact) -> bool:
    """Accession numbers sort chronologically, so a plain comparison works."""
    if a.accession and b.accession:
        return a.accession < b.accession
    return False


def annual_facts(facts: list[Fact]) -> list[Fact]:
    """Full-year facts only -- the cleanest basis for generated questions."""
    return [f for f in facts if f.fiscal_period == "FY"]


def pick_revenue(facts: list[Fact], fiscal_year: int) -> Fact | None:
    """Revenue is reported under two different tags depending on the filer.

    Preferring the contract-with-customer tag matches how modern filings
    present it; falling back to `Revenues` covers the rest. Getting this wrong
    produces a golden-set question with no answer, which then looks like a
    retrieval failure rather than a data bug.
    """
    preferred = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
    ]
    for tag in preferred:
        for f in facts:
            if f.tag == tag and f.fiscal_year == fiscal_year and f.fiscal_period == "FY":
                return f
    return None


def to_rows(facts: list[Fact]) -> list[tuple]:
    """Tuples in the column order used by the xbrl_facts INSERT."""
    return [
        (
            f.cik,
            f.taxonomy,
            f.tag,
            f.unit,
            f.fiscal_year,
            f.fiscal_period,
            f.period_start,
            f.period_end,
            f.value,
            f.accession,
            f.form,
        )
        for f in facts
    ]


def format_value(value: float, unit: str) -> str:
    """Human-readable rendering used in golden-set expected answers."""
    if unit == "USD":
        if abs(value) >= 1_000_000_000:
            return f"${value / 1_000_000_000:,.2f} billion"
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:,.1f} million"
        return f"${value:,.2f}"
    if unit == "shares":
        return f"{value:,.0f} shares"
    if unit.startswith("USD/"):
        return f"${value:,.2f}"
    return f"{value:,.2f} {unit}"
