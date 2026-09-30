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
    "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost": "research and development expense",
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


# A 52/53-week year "ending on the Sunday nearest December 31" ends as late as
# January 3. A year meant to end in January ends at the other end of the month:
# NVIDIA's ends on the last Sunday of January, never before the 25th. A week is
# a margin on the first and nowhere near the second.
JANUARY_SPILLOVER_DAYS = 7


def fiscal_year_ending(period_end: date) -> int:
    """The fiscal year named by a period ending on `period_end`.

    The rule, and why it has an exception, is in `fiscal_year_of`. This is the
    same rule for callers that have a date and no `fy` -- the filing index.
    """
    if period_end.month == 1 and period_end.day <= JANUARY_SPILLOVER_DAYS:
        return period_end.year - 1
    return period_end.year


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

    The period the fact actually covers lives in `start`/`end`, and the fiscal
    year is the calendar year of `end` -- except when `end` falls in the first
    `JANUARY_SPILLOVER_DAYS` of January. Then it is the year before.

    That exception was learned the same way. Johnson & Johnson reports a
    52/53-week year ending on the Sunday nearest December 31, so some of its
    years end in early January: fiscal 2022 ran 2022-01-03 -> 2023-01-01. Taking
    the year of `end` filed it under 2023, where it collided with the real
    fiscal 2023 and, as the earlier accession, displaced it. Fiscal 2020 (ending
    2021-01-03) was filed under 2021, fiscal 2021 under 2022, and nothing under
    2020. Twelve golden cases asked about JNJ's 2023 and were graded against
    fiscal-2022 figures (D13 in docs/METRICS.md). The rule it replaced said no
    filer in this universe ended a year in early January; JNJ had done so eight
    times.

    A real January year-end sits at the other end of the month. NVIDIA's fiscal
    2024 ended 2024-01-28 and NVIDIA calls it fiscal 2024, so late January keeps
    the year of `end`.

    Checked on 29 Sep 2026 against the `fy` of each 10-K for its own year, for
    all 135 fiscal years of the eight companies that have one in
    `companyfacts`. The rule agrees on 131. That includes all eight
    early-January JNJ year-ends, where the old rule was wrong on each. The four
    disagreements are NVIDIA's 10-Ks for fiscal 2011-2014. Their `fy` is
    internally inconsistent: two consecutive years both claim 2010, and no
    filing claims 2014. The rule gives the unbroken sequence.

    Where this still breaks: a filer whose year ends in late January or early
    February and is named for the year it *starts* in, as many retailers do.
    None are in this universe. A mislabel that shifts only some years, as JNJ's
    did, leaves a gap or two adjacent years that are not a year apart, and
    `ingest verify-facts` flags both. One that shifts every year alike leaves
    neither. Only the filer's own `fy` shows it, which is why the rule was
    checked against `fy` above.
    """
    return fiscal_year_ending(period_end) if period_end is not None else reported_fy


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


def misdated_years(facts: list[Fact]) -> list[tuple]:
    """Consecutive years of one series whose labels and periods disagree.

    Two years labelled n apart must end n years apart. This checks the labels
    against the calendar without knowing the fiscal-year rule, which is the
    point: a check that derives years the same way as the ingest agrees with
    whatever the ingest got wrong. JNJ under the old rule failed it both ways.
    "2023" (ending 2023-01-01) and "2024" (ending 2024-12-29) were labelled
    adjacent but ended two years apart, because real fiscal 2023 had been
    displaced. "2019" and "2021" were labelled two apart but ended one year
    apart.

    What it cannot see is every year shifted by the same amount. Nothing in the
    database can, because the filing index derives its year by the same rule.
    That takes the filer's own `fy` (see `fiscal_year_of`).

    Returns (cik, tag, unit, year, period_end, next_year, next_period_end).
    """
    series: dict[tuple, list[Fact]] = {}
    for f in facts:
        if f.fiscal_period == "FY" and f.period_end is not None:
            series.setdefault((f.cik, f.taxonomy, f.tag, f.unit), []).append(f)

    out = []
    for (cik, _, tag, unit), rows in series.items():
        rows.sort(key=lambda f: f.fiscal_year)
        for a, b in zip(rows, rows[1:], strict=False):
            years = b.fiscal_year - a.fiscal_year
            days = (b.period_end - a.period_end).days
            if not years * FULL_YEAR_MIN_DAYS <= days <= years * FULL_YEAR_MAX_DAYS:
                out.append(
                    (cik, tag, unit, a.fiscal_year, a.period_end, b.fiscal_year, b.period_end)
                )
    return sorted(out)


def missing_years(facts: list[Fact]) -> list[tuple[str, int]]:
    """(cik, fiscal_year) with no annual fact at all, inside a company's range.

    One tag can legitimately skip a year -- filers switch concepts -- but a
    company does not skip a fiscal year. A hole in every tag at once means a
    year's facts were filed under a neighbouring label. Under the old rule JNJ
    had three: 2009, 2015 and 2020.
    """
    years: dict[str, set[int]] = {}
    for f in facts:
        if f.fiscal_period == "FY":
            years.setdefault(f.cik, set()).add(f.fiscal_year)
    return sorted(
        (cik, y)
        for cik, present in years.items()
        for y in range(min(present), max(present) + 1)
        if y not in present
    )


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

def pick_research_and_development(
    facts: list[Fact], fiscal_year: int
) -> Fact | None:
    """R&D may be reported under different XBRL tags depending on the filer.

    Prefer the broader operating R&D line that excludes acquired in-process
    R&D when available, then fall back to the standard R&D tag.
    """
    preferred = [
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        "ResearchAndDevelopmentExpense",
    ]

    for tag in preferred:
        for f in facts:
            if (
                f.tag == tag
                and f.fiscal_year == fiscal_year
                and f.fiscal_period == "FY"
            ):
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
