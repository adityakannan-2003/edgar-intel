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


# A fiscal year is not exactly 365 days. 52/53-week filers (AAPL, COST) land
# anywhere in this band, and a leap year adds one more. Anything outside it is
# a quarter, a half, or a nine-month stub -- not a full year.
FULL_YEAR_MIN_DAYS = 340
FULL_YEAR_MAX_DAYS = 380


def covers_full_year(period_start: date | None, period_end: date | None) -> bool:
    """Whether a duration fact spans an entire fiscal year."""
    if period_start is None or period_end is None:
        return False
    return FULL_YEAR_MIN_DAYS <= (period_end - period_start).days <= FULL_YEAR_MAX_DAYS


def fiscal_year_of(period_end: date | None, reported_fy: int) -> int:
    """The fiscal year a fact BELONGS TO -- not the one it was reported in.

    This distinction is the single most expensive thing to get wrong about
    `companyfacts`, because getting it wrong produces no error.

    Every entry carries `fy` and `fp`. They look like the fact's period and are
    not: they identify **the report the fact appeared in**. A 10-K shows three
    years of income statement, so Caterpillar's FY2025 10-K emits FY2025,
    FY2024 *and* FY2023 revenue -- all three stamped `fy: 2025, fp: "FY"`.
    Keying on `fy` therefore files three different values under one fiscal year
    and, across three ingested filings, gives the same year several conflicting
    values depending on which filing is read.

    That is exactly what happened here. The golden set contained
    `num-CAT-Revenues-2023` and `num-CAT-Revenues-2024` with the identical
    expected value, and `num-CAT-ResearchAndDevelopmentExpense-2025` twice with
    two different expected values. Every answer graded against the wrong year's
    figure counted as a model failure. The model was frequently right.

    The period the fact actually covers lives in `start`/`end`. The year of
    `end` is the fiscal year for every filer in this universe, including the
    January enders (NVIDIA's fiscal 2024 ended 2024-01-28 and NVIDIA calls it
    fiscal 2024).

    Where this rule breaks: a filer whose fiscal year ends in the first weeks
    of January and labels it by the *starting* calendar year. None are in this
    universe, and `ingest verify-facts` cross-checks every derived year against
    the period end recorded in `filings`, so the assumption is tested rather
    than trusted.
    """
    return period_end.year if period_end is not None else reported_fy


def extract_facts(
    cik: str,
    company_facts: dict[str, Any],
    tags: dict[str, str] | None = None,
    forms: tuple[str, ...] = ("10-K",),
) -> list[Fact]:
    """Flatten companyfacts JSON into one Fact per (tag, unit, fiscal year).

    Three subtleties, each of which silently corrupts the ground truth:

    1. `fy`/`fp` describe the *report*, not the fact. See `fiscal_year_of`.
    2. The same (tag, period) appears in several filings -- once as the current
       year and again as a comparative. We keep the **earliest** accession, so
       "FY2023 revenue" resolves to what the FY2023 10-K reported rather than a
       later restatement. That is not a preference for old data: the retrieval
       corpus is the filing text, and the FY2023 filing contains the original
       figure. Grading against a restatement would mark correct retrieval wrong.
    3. Instantaneous facts (balance-sheet items) have no `start`, so their
       duration cannot be checked. A 10-K also carries quarter-end balances, and
       those must not be filed as annual. They are kept only when their date is
       a fiscal year-end -- established from the duration facts of this same
       company, so the filter is derived from the data rather than guessed.
    """
    tags = tags or CORE_TAGS
    facts_root = company_facts.get("facts", {})

    raw: list[tuple[str, str, str, dict[str, Any]]] = []
    for taxonomy in ("us-gaap", "dei"):
        tax_block = facts_root.get(taxonomy, {})
        for tag, tag_block in tax_block.items():
            if tag not in tags:
                continue
            for unit, entries in (tag_block.get("units") or {}).items():
                for entry in entries:
                    if forms and entry.get("form") not in forms:
                        continue
                    if entry.get("val") is None or entry.get("fy") is None:
                        continue
                    raw.append((taxonomy, tag, unit, entry))

    # Pass 1: the company's fiscal year-end dates, taken from the facts that
    # can prove they span a year.
    year_ends = {
        _parse_date(e.get("end"))
        for _, _, _, e in raw
        if covers_full_year(_parse_date(e.get("start")), _parse_date(e.get("end")))
    }
    year_ends.discard(None)

    # Pass 2: keep annual facts only, keyed by the year they describe.
    out: dict[tuple, Fact] = {}
    for taxonomy, tag, unit, entry in raw:
        period_start = _parse_date(entry.get("start"))
        period_end = _parse_date(entry.get("end"))
        if period_end is None:
            continue

        if period_start is not None:
            if not covers_full_year(period_start, period_end):
                continue
        elif year_ends and period_end not in year_ends:
            # Instantaneous fact dated at a quarter end, not a year end.
            continue

        fiscal_year = fiscal_year_of(period_end, int(entry["fy"]))
        key = (taxonomy, tag, unit, fiscal_year)
        candidate = Fact(
            cik=cik,
            taxonomy=taxonomy,
            tag=tag,
            unit=unit,
            fiscal_year=fiscal_year,
            fiscal_period="FY",
            period_start=period_start,
            period_end=period_end,
            value=float(entry["val"]),
            accession=entry.get("accn"),
            form=entry.get("form"),
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


def conflicting_years(facts: list[Fact]) -> list[tuple]:
    """(cik, tag, unit, fiscal_year) groups that still hold more than one value.

    After the fix this must be empty. It is kept as an assertion rather than
    deleted, because the failure mode it guards is invisible: a second value for
    a year does not raise, it just makes half the golden set expect the wrong
    number.
    """
    groups: dict[tuple, set[float]] = {}
    for f in facts:
        groups.setdefault((f.cik, f.tag, f.unit, f.fiscal_year), set()).add(f.value)
    return sorted(k for k, v in groups.items() if len(v) > 1)


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
