"""Golden-set construction.

Three kinds of case, in descending order of how much you can trust them:

  numeric single-hop   Generated directly from an XBRL fact. The expected value
                       came from the SEC, not from a model. Graded by numeric
                       comparison within a relative tolerance.

  numeric comparative  Derived from two XBRL facts (year over year, or a ratio).
                       Still fully verifiable, and much harder for the system --
                       it needs two pieces of evidence, so it is where
                       single-passage retrieval starts to fail.

  narrative            Hand-written questions over Risk Factors and MD&A, with a
                       human-written reference answer. Graded by LLM judge,
                       which is only trustworthy to the extent the judge agrees
                       with humans -- hence judge.py.

Generating the numeric cases automatically is what makes a 200-case set
achievable for one person in an afternoon. Hand-writing 200 questions is what
stops most people from ever building an eval set at all.
"""

from __future__ import annotations

import json
import random
from typing import Any

from .. import db
from ..ingest.xbrl import CORE_TAGS, format_value
from .schemas import EvalCase


def _fmt(value: float) -> str:
    """Render a fact for the human-readable note.

    Adaptive precision because the same code path handles both 383,285,000,000
    and 1.43: a fixed '%,.0f' turns every per-share figure into "1" and makes
    the note actively misleading.
    """
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


QUESTION_TEMPLATES: dict[str, list[str]] = {
    "single": [
        "What was {company}'s {label} for fiscal year {year}?",
        "How much {label} did {company} report in FY{year}?",
        "According to {company}'s FY{year} 10-K, what was {label}?",
    ],
    "yoy": [
        "How did {company}'s {label} change from FY{prev} to FY{year}?",
        "What was the year-over-year change in {company}'s {label} between FY{prev} and FY{year}?",
    ],
    "ratio": [
        "What was {company}'s {numerator} as a percentage of {denominator} in FY{year}?",
    ],
}


def _companies() -> dict[str, dict[str, Any]]:
    rows = db.query("SELECT cik, ticker, name FROM companies ORDER BY ticker")
    return {r["cik"]: r for r in rows}


def _covered_periods() -> set[tuple[str, int]]:
    """(cik, fiscal_year) pairs that have an ingested filing.

    This is the constraint that makes the generated questions valid.
    `companyfacts` returns a company's entire XBRL history -- fifteen years for
    a mature filer -- while ingestion fetches only the most recent few filings.
    Generating a question for FY2011 when no FY2011 filing was ever ingested
    produces a case the retrieval pipeline cannot answer no matter how good it
    is, and a set full of those makes recall look broken when it is not.

    The facts themselves are still worth keeping for every year: the agent's
    `get_fact` and `compare_fact` tools read XBRL directly and can answer about
    years whose filing text is absent. It is only the *retrieval* evaluation
    that needs the filing present.
    """
    rows = db.query(
        "SELECT DISTINCT cik, fiscal_year FROM filings WHERE fiscal_year IS NOT NULL"
    )
    return {(r["cik"], int(r["fiscal_year"])) for r in rows}


def filter_to_covered(
    facts: list[dict[str, Any]], covered: set[tuple[str, int]]
) -> list[dict[str, Any]]:
    """Keep only facts whose (cik, fiscal_year) has an ingested filing."""
    return [f for f in facts if (f["cik"], int(f["fiscal_year"])) in covered]


def _facts(cik: str | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT cik, tag, unit, fiscal_year, fiscal_period, value
          FROM xbrl_facts
         WHERE fiscal_period = 'FY'
    """
    params: list[Any] = []
    if cik:
        sql += " AND cik = %s"
        params.append(cik)
    sql += " ORDER BY cik, tag, fiscal_year"
    return db.query(sql, params)

PREFERRED_TAG_FAMILIES: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
    ],
    "research_and_development": [
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        "ResearchAndDevelopmentExpense",
    ],
}


def _canonicalize_fact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose one preferred XBRL tag for equivalent financial concepts.

    Some filers report the same economic concept under different XBRL tags.
    Without canonicalization, the golden set can generate duplicate questions
    for the same company/year with conflicting expected values.
    """
    family_by_tag = {
        tag: family
        for family, tags in PREFERRED_TAG_FAMILIES.items()
        for tag in tags
    }

    preference = {
        tag: rank
        for tags in PREFERRED_TAG_FAMILIES.values()
        for rank, tag in enumerate(tags)
    }

    selected: dict[tuple[str, int], dict[str, Any]] = {}
    untouched: list[dict[str, Any]] = []

    for row in rows:
        tag = row["tag"]
        family = family_by_tag.get(tag)

        if family is None:
            untouched.append(row)
            continue

        key = (family, int(row["fiscal_year"]))
        current = selected.get(key)

        if current is None or preference[tag] < preference[current["tag"]]:
            selected[key] = row

    return untouched + list(selected.values())

def generate_numeric_cases(limit_per_company: int = 20, seed: int = 7) -> list[EvalCase]:
    rng = random.Random(seed)
    companies = _companies()
    covered = _covered_periods()
    facts = filter_to_covered(_facts(), covered)

    by_company: dict[str, list[dict[str, Any]]] = {}
    for f in facts:
        by_company.setdefault(f["cik"], []).append(f)

    cases: list[EvalCase] = []
    for cik, rows in by_company.items():
        rows = _canonicalize_fact_rows(rows)
        company = companies.get(cik, {})
        name = company.get("name") or cik
        ticker = company.get("ticker")

        # ---- single-hop
        pool = [r for r in rows if r["tag"] in CORE_TAGS]
        rng.shuffle(pool)
        taken = 0
        for row in pool:
            if taken >= limit_per_company:
                break
            label = CORE_TAGS[row["tag"]]
            template = rng.choice(QUESTION_TEMPLATES["single"])
            question = template.format(company=name, label=label, year=row["fiscal_year"])
            value = float(row["value"])
            cases.append(
                EvalCase(
                    case_id=f"num-{ticker or cik}-{row['tag']}-{row['fiscal_year']}",
                    kind="numeric",
                    question=question,
                    expected=format_value(value, row["unit"]),
                    expected_value=value,
                    unit=row["unit"],
                    ticker=ticker,
                    fiscal_year=row["fiscal_year"],
                    tag=row["tag"],
                    difficulty="single_hop",
                )
            )
            taken += 1

        # ---- year-over-year comparatives
        cases.extend(_yoy_cases(rows, name, ticker, cik, rng, 6, covered))

    return cases


def _yoy_cases(
    rows: list[dict[str, Any]],
    name: str,
    ticker: str | None,
    cik: str,
    rng: random.Random,
    max_cases: int = 6,
    covered: set[tuple[str, int]] | None = None,
) -> list[EvalCase]:
    by_tag: dict[str, dict[int, float]] = {}
    for r in rows:
        if r["tag"] not in CORE_TAGS:
            continue
        by_tag.setdefault(r["tag"], {})[int(r["fiscal_year"])] = float(r["value"])

    out: list[EvalCase] = []
    tags = list(by_tag)
    rng.shuffle(tags)
    for tag in tags:
        if len(out) >= max_cases:
            break
        years = sorted(by_tag[tag])
        for prev, cur in zip(years, years[1:], strict=False):
            if cur - prev != 1:
                continue
            # Both sides must be in the corpus: a comparison against a year
            # whose filing is absent is half-unanswerable.
            if covered is not None and (
                (cik, prev) not in covered or (cik, cur) not in covered
            ):
                continue
            a, b = by_tag[tag][prev], by_tag[tag][cur]
            if a == 0:
                continue
            delta_pct = (b - a) / abs(a) * 100
            label = CORE_TAGS[tag]
            question = rng.choice(QUESTION_TEMPLATES["yoy"]).format(
                company=name, label=label, prev=prev, year=cur
            )
            direction = "increased" if delta_pct >= 0 else "decreased"
            out.append(
                EvalCase(
                    case_id=f"yoy-{ticker or cik}-{tag}-{prev}-{cur}",
                    kind="numeric",
                    question=question,
                    expected=f"{direction} {abs(delta_pct):.1f}%",
                    expected_value=round(delta_pct, 4),
                    unit="percent",
                    ticker=ticker,
                    fiscal_year=cur,
                    tag=tag,
                    difficulty="comparative",
                    notes=f"FY{prev}={_fmt(a)}, FY{cur}={_fmt(b)}",
                )
            )
            break
    return out[:max_cases]


# --------------------------------------------------------------- narrative
NARRATIVE_SEEDS: list[dict[str, str]] = [
    {
        "slug": "supply-concentration",
        "item": "1A",
        "question": "What supply chain or supplier concentration risks does {company} disclose?",
        "reference": "The company discloses dependence on a limited number of suppliers "
        "and single-source components, and warns that disruption at those suppliers "
        "could materially affect its ability to meet demand.",
    },
    {
        "slug": "competition",
        "item": "1A",
        "question": "How does {company} characterise competitive pressure in its risk factors?",
        "reference": "The company describes highly competitive markets with rapid "
        "technological change, price pressure, and competitors with greater resources "
        "in some segments.",
    },
    {
        "slug": "fx-exposure",
        "item": "7A",
        "question": "What foreign currency exposure does {company} report and how is it managed?",
        "reference": "The company reports material revenue and costs denominated in "
        "currencies other than the US dollar and describes hedging with derivative "
        "instruments to reduce, but not eliminate, that exposure.",
    },
    {
        "slug": "revenue-drivers",
        "item": "7",
        "question": "What does {company}'s MD&A give as the main drivers of the change in revenue?",
        "reference": "Management attributes the revenue change to a combination of "
        "volume, pricing, product mix, and in some periods foreign exchange effects.",
    },
    {
        "slug": "legal",
        "item": "3",
        "question": "What material legal proceedings does {company} disclose?",
        "reference": "The company describes pending litigation and regulatory matters "
        "and states whether it expects a material adverse effect on its financial position.",
    },
]

NARRATIVE_REFERENCES = {
    "nar-AAPL-legal": (
        "Apple discloses EU Digital Markets Act proceedings, including a €500 million "
        "fine in the Article 5(4) investigation that Apple has appealed and a separate "
        "Article 6(4) investigation. It also discloses the DOJ and state attorneys "
        "general antitrust lawsuit alleging monopolization of smartphone markets, "
        "as well as the Epic Games litigation concerning App Store rules and the "
        "2021 injunction. Apple states that other ordinary-course legal proceedings "
        "remain unresolved and adverse outcomes could materially affect results."
    ),

    "nar-AAPL-supply-concentration": (
        "Apple says a significant majority of its manufacturing is performed by "
        "outsourcing partners, primarily in China mainland, India, Japan, South Korea, "
        "Taiwan and Vietnam. It relies on single or limited sources for many critical "
        "components and primarily Asian partners for final assembly. Trade restrictions, "
        "geopolitical conflict, natural disasters, shortages, supplier failures and "
        "capacity constraints can therefore disrupt supply, increase costs and delay "
        "production, while changing suppliers can be costly and time-consuming."
    ),

    "nar-AAPL-revenue-drivers": (
        "Apple's 2025 net sales increased 6% to $416.2 billion. Growth was driven "
        "primarily by higher Services, iPhone, Mac and iPad sales, partly offset by "
        "lower Wearables, Home and Accessories sales. Services rose 14%, Mac 12%, "
        "iPad 5% and iPhone 4%. Geographically, most regions grew, while Greater "
        "China declined 4%, primarily because of lower iPhone sales."
    ),

    "nar-CAT-fx-exposure": (
        "Caterpillar's Machinery, Power & Energy operations use foreign-currency "
        "forward and option contracts to manage unmatched foreign-currency cash flows, "
        "with policy allowing anticipated exposures to be managed for up to about "
        "five years. Its primary currency exposures include the Australian dollar, "
        "Chinese yuan, euro, Indian rupee and Mexican peso. Financial Products also "
        "uses forwards, options and cross-currency contracts to reduce currency "
        "mismatches between assets, liabilities and future transactions."
    ),

    "nar-CAT-competition": (
        "Caterpillar says it operates in a highly competitive environment and competes "
        "on product performance, customer service, quality and price. Aggressive "
        "competitor pricing or product strategies, failure to price or manufacture "
        "competitively, changes in customer discount expectations, weak pricing "
        "conditions and growth in competitors' rental fleets could reduce Caterpillar's "
        "industry share and put downward pressure on prices and profitability."
    ),

    "nar-COST-revenue-drivers": (
        "Costco's 2025 net sales increased 8% to $269.9 billion. The main driver was "
        "a 6% increase in comparable sales, including about 5% higher shopping "
        "frequency and about 1% higher average ticket, together with sales from 24 "
        "net new warehouses. Core merchandise sales increased 10%. Lower gasoline "
        "prices and unfavorable foreign-exchange movements partially offset the growth."
    ),

    "nar-COST-fx-exposure": (
        "Costco's foreign subsidiaries conduct some transactions in currencies other "
        "than their functional currencies, exposing Costco to exchange-rate movements. "
        "It manages part of this exposure with forward foreign-exchange contracts, "
        "primarily to economically hedge U.S.-dollar merchandise inventory expenditures "
        "of international subsidiaries. Costco states that these contracts are used "
        "for risk mitigation rather than speculative trading."
    ),

    "nar-COST-supply-concentration": (
        "Costco depends on the orderly operation of its merchandise receiving and "
        "distribution network, particularly its depots, as well as processing, "
        "packaging and manufacturing facilities. Disruptions at these facilities or "
        "in supporting systems can impair Costco's ability to obtain and move products "
        "and can reduce sales and member satisfaction. Product availability and "
        "consistent quality are also particularly important for its growing "
        "Kirkland Signature private-label business."
    ),

    "nar-JNJ-competition": (
        "Johnson & Johnson describes its product markets as highly competitive across "
        "both operating segments and all geographic markets. It competes on "
        "cost-effectiveness, technological innovation, intellectual-property rights, "
        "product performance, perceived product advantages, pricing, availability and "
        "reimbursement. It also competes for acquisitions and licensing opportunities, "
        "and more effective or less expensive competing products or faster competitor "
        "development can reduce sales and impair its ability to commercialize products."
    ),

    "nar-JNJ-fx-exposure": (
        "Johnson & Johnson reports substantial international currency exposure, with "
        "about 43% of fiscal 2025 sales occurring outside the United States. Changes "
        "in foreign currencies affect revenues, expenses and translation of overseas "
        "results into U.S. dollars. The company uses financial instruments to mitigate "
        "the cash-flow effects of exchange-rate fluctuations, although unhedged "
        "exposures remain subject to currency movements."
    ),

    "nar-JNJ-supply-concentration": (
        "Johnson & Johnson operates 63 manufacturing facilities and also sources from "
        "thousands of suppliers worldwide. Manufacturing can be disrupted by quality "
        "issues, regulation, labor problems, raw-material shortages, natural disasters "
        "and geopolitical events. The company also relies on third parties for raw "
        "materials, components and finished products, making it dependent on their "
        "capacity, quality, yields, delivery timing and pricing; failure or loss of a "
        "supplier can cause shortages, delays, lost sales and higher costs."
    ),

    "nar-MSFT-competition": (
        "Microsoft says it faces intense competition across all of its markets, from "
        "large global companies with major R&D resources to smaller specialized firms. "
        "Many markets have low barriers to entry and change rapidly. Microsoft also "
        "identifies especially strong competition in AI from hyperscalers, open-source "
        "offerings and frontier-model providers, as well as competition in cloud and "
        "other software business models, requiring continued innovation and adaptation."
    ),

    "nar-MSFT-fx-exposure": (
        "Microsoft has foreign-currency exposure from forecasted transactions, assets "
        "and liabilities. It monitors foreign-currency exposures daily and uses hedges "
        "and derivative instruments to manage the risk. Principal exposures include "
        "the euro, Japanese yen, British pound, Canadian dollar and Australian dollar."
    ),

    "nar-NVDA-revenue-drivers": (
        "NVIDIA's fiscal 2026 revenue increased 65% to $215.938 billion. "
        "Compute & Networking revenue increased 67%, with the year-over-year "
        "increase driven by accelerated computing and AI. Data Center compute "
        "revenue grew 59% on demand for the Blackwell computing platform, while "
        "Data Center networking grew 142% from the ramp of NVLink compute fabric "
        "and growth in Ethernet and InfiniBand platforms. Graphics revenue "
        "increased 57%, driven by Blackwell architecture sales."
    ),

    "nar-NVDA-fx-exposure": (
        "NVIDIA says its direct foreign-exchange exposure is relatively limited because "
        "substantially all sales are denominated in U.S. dollars. It nevertheless uses "
        "foreign-currency forward contracts to offset exchange-rate movements, including "
        "hedges of forecasted foreign-currency expenses and balance-sheet exposures."
    ),

    "nar-PG-revenue-drivers": (
        "P&G's fiscal 2026 net sales increased 3% to $87.0 billion, driven by about "
        "2% favorable foreign exchange and 1% higher pricing, while unit volume and "
        "mix were unchanged overall. Organic sales increased 1%. Results varied by "
        "business segment, with Beauty producing the strongest reported sales growth."
    ),

    "nar-PG-fx-exposure": (
        "P&G is exposed to currency-rate movements because it manufactures, sells and "
        "finances operations globally. It first uses diversification, natural offsets "
        "and centralized exposure netting as operational hedges. For additional risk "
        "management it primarily uses forward contracts and currency swaps, generally "
        "with maturities under 18 months, and monitors derivative exposure using market "
        "valuation, sensitivity analysis and value-at-risk techniques."
    ),

    "nar-PG-legal": (
        "P&G discloses an unresolved U.K. Environment Agency matter involving its U.K. "
        "subsidiary's prior inadvertent failure to obtain a required emissions permit "
        "for a London manufacturing site. The site has been registered since March "
        "2021, and in July 2025 the agency indicated an intended civil penalty of less "
        "than $2 million. P&G states there were no other matters required to be "
        "disclosed under Item 3 for the period."
    ),

    "nar-UNH-competition": (
        "UnitedHealth says its businesses face significant competition in all markets "
        "and that some competitors may have advantages in particular geographies or "
        "product areas. Industry consolidation can make it harder to retain customers, "
        "obtain favorable supplier terms and maintain profitability. Competitiveness "
        "also depends on innovation, value-based care models, technology and analytics, "
        "and maintaining relationships with physicians and other health-care providers."
    ),

    "nar-UNH-supply-concentration": (
        "UnitedHealth's risk is primarily dependence on third-party health-care and "
        "technology relationships rather than a traditional manufacturing supply chain. "
        "It relies on physicians, hospitals, pharmaceutical and other service providers "
        "and, in some circumstances, third-party vendors that process, store and "
        "transmit large amounts of data. Vendor failures, provider departures, contract "
        "disputes or failures outside UnitedHealth's direct control can disrupt services "
        "and adversely affect revenue, operations and customer relationships."
    ),

        "nar-CAT-revenue-drivers": (
        "Caterpillar's 2025 sales and revenues increased 4% to $67.589 billion "
        "from $64.809 billion in 2024. The increase was primarily driven by "
        "$3.389 billion of higher sales volume, mainly from higher equipment "
        "sales to end users, partially offset by $817 million of unfavorable "
        "price realization. Currency and higher Financial Products revenues "
        "also contributed modestly."
    ),

    "nar-MSFT-legal": (
        "Microsoft discloses an Irish Data Protection Commission matter involving "
        "LinkedIn's targeted-advertising practices under the GDPR. The regulator "
        "issued a final decision and fine in October 2024, which LinkedIn appealed "
        "in November 2024; a preliminary hearing was held in December 2025. "
        "Microsoft also reports other ordinary-course claims and suits. As of "
        "June 30, 2026, it had accrued $553 million of legal liabilities and "
        "estimated that adverse outcomes of approximately $400 million beyond "
        "recorded amounts were reasonably possible."
    ),

    "nar-NVDA-legal": (
        "NVIDIA discloses securities class-action and derivative lawsuits alleging "
        "false or misleading statements concerning channel inventory and the effect "
        "of cryptocurrency mining on GPU demand in 2017 and 2018. Related derivative "
        "actions allege claims including breach of fiduciary duty and insider trading. "
        "As of January 25, 2026, NVIDIA had not accrued contingent liabilities for "
        "these proceedings because losses were considered reasonably possible but "
        "not probable, and a possible loss or range of loss could not be reasonably "
        "estimated."
    ),

    "nar-UNH-legal": (
        "UnitedHealth Group discloses a variety of legal actions and regulatory "
        "inquiries involving matters such as health benefit administration, medical "
        "malpractice, employment, intellectual property, antitrust, privacy and "
        "contract claims. It also describes government investigations and audits, "
        "including Medicare risk-adjustment reviews. A DOJ False Claims Act case "
        "alleges improper risk-adjustment submissions; in March 2025 a Special Master "
        "recommended summary judgment for UnitedHealth on the remaining claims, and "
        "in April 2025 the DOJ asked the court to reject that recommendation. "
        "UnitedHealth states that it cannot reasonably estimate the outcome."
    ),
}

def generate_narrative_cases(max_per_company: int = 3, seed: int = 7) -> list[EvalCase]:
    """Templated narrative questions seeded per company.

    These are a starting point, not a finished set. The reference answers are
    deliberately generic; the discipline is to replace each one with what the
    filing actually says after reading it. A narrative eval set you have not
    read is a narrative eval set you cannot defend.
    """
    rng = random.Random(seed)
    cases: list[EvalCase] = []
    for row in db.query("SELECT cik, ticker, name FROM companies ORDER BY ticker"):
        seeds = NARRATIVE_SEEDS[:]
        rng.shuffle(seeds)
        for seed_case in seeds[:max_per_company]:
            year_row = db.query_one(
                "SELECT MAX(fiscal_year) AS fy FROM filings WHERE cik = %s", (row["cik"],)
            )
            fy = year_row["fy"] if year_row else None
            case_id = f"nar-{row['ticker'] or row['cik']}-{seed_case['slug']}"

            reference = NARRATIVE_REFERENCES.get(case_id)
            if reference is None:
                raise ValueError(f"Missing narrative reference for {case_id}")

            cases.append(
                    EvalCase(
                    case_id=case_id,
                    kind="narrative",
                    question=NARRATIVE_QUESTION_OVERRIDES.get(
                    case_id,
                    seed_case["question"].format(company=row["name"]),
                    ),
                    expected=reference,
                    ticker=row["ticker"],
                    fiscal_year=fy,
                    difficulty="single_hop",
                    notes=f"human-reviewed reference grounded in FY{fy} filing; "
                        f"expected evidence in Item {seed_case['item']}",
                )
            )
    return cases

NARRATIVE_QUESTION_OVERRIDES = {
    "nar-UNH-supply-concentration": (
        "UnitedHealth says noncompliance with privacy and security requirements, "
        "or a privacy or security breach involving the company or one of its "
        "third-party service providers, could harm its reputation and business. "
        "Consequences can include mandatory disclosure, loss of existing or new "
        "customers, increased incident-management and remediation costs, and "
        "significant fines, penalties and litigation awards."
    ),
}

def test_all_narrative_cases_have_human_references():
    from edgar_intel.evals.goldenset import (
        NARRATIVE_REFERENCES,
        generate_narrative_cases,
    )

    cases = generate_narrative_cases()

    assert len(cases) == 24

    for case in cases:
        assert case.case_id in NARRATIVE_REFERENCES
        assert "REPLACE this reference" not in case.expected
        assert len(case.expected) > 80

# ----------------------------------------------------------- evidence linking
import re

_NARRATIVE_STOPWORDS = {
    "about", "after", "also", "because", "been", "being", "company",
    "could", "does", "during", "fiscal", "from", "have", "into",
    "more", "other", "report", "reports", "reported", "says",
    "states", "that", "their", "these", "they", "this", "through",
    "under", "uses", "using", "were", "which", "while", "with",
    "year",
}


def _narrative_reference_terms(text: str) -> list[str]:
    """Return distinctive reference-answer terms for evidence labeling."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())

    terms: list[str] = []
    for token in tokens:
        if len(token) < 4:
            continue
        if token.isdigit():
            continue
        if token in _NARRATIVE_STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)

    return terms[:24]

NARRATIVE_EVIDENCE_TERMS = {
    "nar-CAT-revenue-drivers": [
        "67,589",
        "64,809",
        "sales volume",
    ],
    "nar-NVDA-revenue-drivers": [
        "215,938",
        "accelerated computing and AI",
        "Blackwell",
    ],
    "nar-PG-revenue-drivers": [
        "87.0 billion",
        "favorable foreign exchange of 2%",
        "pricing of 1%",
    ],
    "nar-UNH-supply-concentration": [
        "third-party service providers",
        "privacy or security breach",
    ],
}

def _numeric_value_needles(value: float) -> list[str]:
    """Generate filing-style representations of a numeric source value."""
    raw = abs(value)
    needles: list[str] = []

    for scale in (1, 1_000, 1_000_000, 1_000_000_000):
        scaled = raw / scale

        if scaled >= 1:
            needles.append(f"{scaled:,.0f}")

    return list(dict.fromkeys(needles))


def _comparative_source_facts(notes: str) -> list[tuple[int, float]]:
    """Parse source facts such as:
    'FY2023=2,108,000,000, FY2024=2,107,000,000'
    """
    matches = re.findall(
        r"FY(\d{4})=([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)",
        notes or "",
    )

    return [
        (int(year), float(value.replace(",", "")))
        for year, value in matches
    ]

def link_evidence(
    cases: list[EvalCase],
    strategy: str,
    k: int = 5,
) -> list[EvalCase]:
    """Attach independently constructed relevant chunk ids.

    Numeric cases are linked using their known XBRL value.

    Narrative cases are linked using distinctive terms from the human-reviewed
    reference answer. We intentionally do not search using the question itself,
    because doing that would make the retriever help construct its own relevance
    labels and artificially inflate retrieval metrics.
    """
    for case in cases:

        # ------------------------------------------------ narrative
        
        if case.kind == "narrative":
            terms_override = NARRATIVE_EVIDENCE_TERMS.get(case.case_id)

            if terms_override:
                clauses = " AND ".join(["c.body ILIKE %s"] * len(terms_override))

                sql = f"""
                    SELECT c.id::text AS id
                    FROM chunks c
                    JOIN filings f ON f.id = c.filing_id
                    JOIN companies co ON co.cik = f.cik
                    WHERE c.strategy = %s
                    AND co.ticker = %s
                    AND f.fiscal_year = %s
                    AND {clauses}
                    LIMIT %s
                """

                params = [
                    strategy,
                    case.ticker,
                    case.fiscal_year,
                    *[f"%{term}%" for term in terms_override],
                    k,
                ]

                rows = db.query(sql, params)
                case.relevant_chunk_ids = [r["id"] for r in rows]
                continue

            terms = _narrative_reference_terms(case.expected)

            if not terms:
                case.relevant_chunk_ids = []
                continue

            tsquery = " | ".join(terms)

            rows = db.query(
                """
                WITH q AS (
                    SELECT to_tsquery('english', %s) AS query
                )
                SELECT
                    c.id::text AS id,
                    ts_rank_cd(
                        to_tsvector('english', c.body),
                        q.query
                    ) AS score
                FROM chunks c
                JOIN filings f
                  ON f.id = c.filing_id
                JOIN companies co
                  ON co.cik = f.cik
                CROSS JOIN q
                WHERE c.strategy = %s
                  AND (%s::text IS NULL OR co.ticker = %s::text)
                  AND (%s::integer IS NULL OR f.fiscal_year = %s::integer)
                  AND to_tsvector('english', c.body) @@ q.query
                ORDER BY score DESC, c.id
                LIMIT %s
                """,
                (
                    tsquery,
                    strategy,
                    case.ticker,
                    case.ticker,
                    case.fiscal_year,
                    case.fiscal_year,
                    k,
                ),
            )

            case.relevant_chunk_ids = [r["id"] for r in rows]
            continue


     # ------------------------------------------------ numeric

        # Comparative / YoY questions
        if case.difficulty == "comparative":
            source_facts = _comparative_source_facts(case.notes)

            if len(source_facts) >= 2:
                source_years = [year for year, _ in source_facts]
                source_values = [value for _, value in source_facts]

                first_needles = _numeric_value_needles(source_values[0])
                second_needles = _numeric_value_needles(source_values[1])

                chunk_ids: list[str] = []

                # First try to find a chunk containing BOTH source values.
                # For a 2023→2024 comparison, prefer the 2024 filing or later.
                for first in first_needles:
                    for second in second_needles:
                        rows = db.query(
                            """
                            SELECT
                                c.id::text AS id,
                                f.fiscal_year
                            FROM chunks c
                            JOIN filings f
                            ON f.id = c.filing_id
                            JOIN companies co
                            ON co.cik = f.cik
                            WHERE c.strategy = %s
                            AND co.ticker = %s
                            AND f.fiscal_year >= %s
                            AND c.body ILIKE %s
                            AND c.body ILIKE %s
                            ORDER BY
                                f.fiscal_year ASC,
                                c.id
                            LIMIT %s
                            """,
                            (
                                strategy,
                                case.ticker,
                                max(source_years),
                                f"%{first}%",
                                f"%{second}%",
                                k,
                            ),
                        )

                        if rows:
                            chunk_ids.extend(r["id"] for r in rows)
                            break

                    if chunk_ids:
                        break

                # Fallback: find evidence for each source value separately.
                if not chunk_ids:
                    for source_year, source_value in source_facts:
                        for needle in _numeric_value_needles(source_value):
                            rows = db.query(
                                """
                                SELECT
                                    c.id::text AS id,
                                    f.fiscal_year
                                FROM chunks c
                                JOIN filings f
                                ON f.id = c.filing_id
                                JOIN companies co
                                ON co.cik = f.cik
                                WHERE c.strategy = %s
                                AND co.ticker = %s
                                AND f.fiscal_year >= %s
                                AND c.body ILIKE %s
                                ORDER BY
                                    CASE
                                        WHEN f.fiscal_year = %s THEN 0
                                        ELSE 1
                                    END,
                                    f.fiscal_year ASC,
                                    c.id
                                LIMIT %s
                                """,
                                (
                                    strategy,
                                    case.ticker,
                                    source_year,
                                    f"%{needle}%",
                                    source_year,
                                    k,
                                ),
                            )

                            if rows:
                                chunk_ids.extend(r["id"] for r in rows)
                                break

                case.relevant_chunk_ids = list(dict.fromkeys(chunk_ids))[:k]
                continue


        # Single-hop numeric questions
        chunk_ids: list[str] = []

        if case.expected_value is not None:
            for needle in _numeric_value_needles(case.expected_value):
                rows = db.query(
                    """
                    SELECT
                        c.id::text AS id,
                        f.fiscal_year
                    FROM chunks c
                    JOIN filings f
                    ON f.id = c.filing_id
                    JOIN companies co
                    ON co.cik = f.cik
                    WHERE c.strategy = %s
                    AND co.ticker = %s
                    AND f.fiscal_year >= %s
                    AND c.body ILIKE %s
                    ORDER BY
                        CASE
                            WHEN f.fiscal_year = %s THEN 0
                            ELSE 1
                        END,
                        f.fiscal_year ASC,
                        c.id
                    LIMIT %s
                    """,
                    (
                        strategy,
                        case.ticker,
                        case.fiscal_year,
                        f"%{needle}%",
                        case.fiscal_year,
                        k,
                    ),
                )

                if rows:
                    chunk_ids.extend(r["id"] for r in rows)
                    break

        case.relevant_chunk_ids = list(dict.fromkeys(chunk_ids))[:k]

    return cases


# ------------------------------------------------------------------- storage
def save(cases: list[EvalCase], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([c.as_dict() for c in cases], fh, indent=2)


def load(path: str) -> list[EvalCase]:
    with open(path, encoding="utf-8") as fh:
        return [EvalCase(**row) for row in json.load(fh)]


def build(
    path: str = "evalset/golden.json",
    numeric_per_company: int = 20,
    narrative_per_company: int = 3,
    strategy: str | None = None,
) -> list[EvalCase]:
    from ..config import get_settings

    strategy = strategy or get_settings().default_strategy
    cases = generate_numeric_cases(numeric_per_company)
    cases += generate_narrative_cases(narrative_per_company)
    cases = link_evidence(cases, strategy)
    save(cases, path)
    return cases
