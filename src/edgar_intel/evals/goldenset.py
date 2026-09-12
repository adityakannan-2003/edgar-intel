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
            cases.append(
                EvalCase(
                    case_id=f"nar-{row['ticker'] or row['cik']}-{seed_case['slug']}",
                    kind="narrative",
                    question=seed_case["question"].format(company=row["name"]),
                    expected=seed_case["reference"],
                    ticker=row["ticker"],
                    fiscal_year=fy,
                    difficulty="single_hop",
                    notes=f"expected evidence in Item {seed_case['item']}; "
                    "REPLACE this reference with what the filing actually says",
                )
            )
    return cases


# ----------------------------------------------------------- evidence linking
def link_evidence(cases: list[EvalCase], strategy: str, k: int = 5) -> list[EvalCase]:
    """Attach relevant chunk ids so retrieval metrics can be computed.

    Evidence is found by lexical search for the *expected answer's* distinctive
    tokens -- not by searching the question. Searching the question would label
    whatever the retriever already returns as "relevant", which makes recall@k
    trivially 1.0 and the metric meaningless. This is the single easiest way to
    build an eval set that flatters your system, and worth saying out loud.
    """
    for case in cases:
        needles: list[str] = []
        if case.kind == "numeric" and case.expected_value is not None:
            raw = abs(case.expected_value)
            # Filings print figures in thousands or millions; try both scales.
            for scale in (1, 1_000, 1_000_000):
                scaled = raw / scale
                if scaled >= 1:
                    needles.append(f"{scaled:,.0f}")
        if case.ticker:
            needles.append(case.ticker)

        chunk_ids: list[str] = []
        for needle in needles:
            rows = db.query(
                """
                SELECT c.id::text AS id
                  FROM chunks c
                  JOIN filings f ON f.id = c.filing_id
                  JOIN companies co ON co.cik = f.cik
                 WHERE c.strategy = %s
                   AND (%s::text IS NULL OR co.ticker = %s::text)
                   AND (%s::integer IS NULL OR f.fiscal_year = %s::integer)
                   AND c.body ILIKE %s
                 LIMIT %s
                """,
                (
                    strategy,
                    case.ticker,
                    case.ticker,
                    case.fiscal_year,
                    case.fiscal_year,
                    f"%{needle}%",
                    k,
                ),
            )
            chunk_ids.extend(r["id"] for r in rows)
            if chunk_ids:
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
