"""Ingestion orchestration: EDGAR -> Postgres.

Idempotent by design. Every insert is an upsert keyed on a natural key
(accession for filings, the full period tuple for XBRL facts), so re-running
after a partial failure costs bandwidth and nothing else. Ingestion jobs fail
halfway more often than people plan for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .. import db
from ..config import get_settings
from .edgar_client import EdgarClient, recent_filings
from .parse import parse_filing
from .xbrl import extract_facts, to_rows


@dataclass(slots=True)
class IngestStats:
    companies: int = 0
    filings: int = 0
    sections: int = 0
    facts: int = 0
    skipped: int = 0
    errors: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "companies": self.companies,
            "filings": self.filings,
            "sections": self.sections,
            "facts": self.facts,
            "skipped": self.skipped,
            "errors": self.errors or [],
        }


def upsert_company(cik: str, ticker: str | None, name: str, sic: str | None = None) -> None:
    db.execute(
        """
        INSERT INTO companies (cik, ticker, name, sic)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (cik) DO UPDATE
           SET ticker = EXCLUDED.ticker,
               name   = EXCLUDED.name,
               sic    = COALESCE(EXCLUDED.sic, companies.sic)
        """,
        (cik, ticker, name, sic),
    )


def upsert_filing(cik: str, row: dict[str, Any], source_url: str) -> int | None:
    fiscal_year = None
    if row.get("period_end"):
        fiscal_year = int(str(row["period_end"])[:4])
    result = db.query_one(
        """
        INSERT INTO filings (cik, accession, form, filing_date, period_end,
                             fiscal_year, primary_doc, source_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (accession) DO UPDATE
           SET period_end  = EXCLUDED.period_end,
               fiscal_year = EXCLUDED.fiscal_year,
               source_url  = EXCLUDED.source_url
        RETURNING id
        """,
        (
            cik,
            row["accession"],
            row["form"],
            row["filing_date"],
            row["period_end"],
            fiscal_year,
            row["primary_doc"],
            source_url,
        ),
    )
    return result["id"] if result else None


def replace_sections(filing_id: int, sections: list[Any]) -> int:
    """Sections are derived data. Replacing wholesale keeps the parser free to
    improve without leaving orphaned rows from an older parse behind."""
    db.execute("DELETE FROM sections WHERE filing_id = %s", (filing_id,))
    rows = [
        (filing_id, s.item, s.title, s.ordinal, s.char_len, s.body)
        for s in sections
    ]
    db.execute_many(
        """
        INSERT INTO sections (filing_id, item, title, ordinal, char_len, body)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        rows,
    )
    return len(rows)


def upsert_facts(facts: list[Any]) -> int:
    rows = to_rows(facts)
    if not rows:
        return 0
    db.execute_many(
        """
        INSERT INTO xbrl_facts (cik, taxonomy, tag, unit, fiscal_year, fiscal_period,
                                period_start, period_end, value, accession, form)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (cik, taxonomy, tag, unit, fiscal_year, fiscal_period, period_end)
        DO NOTHING
        """,
        rows,
    )
    return len(rows)


def ingest_universe(
    tickers: list[str] | None = None,
    forms: list[str] | None = None,
    years_back: int | None = None,
    progress=None,
) -> IngestStats:
    s = get_settings()
    tickers = [t.upper() for t in (tickers or s.universe)]
    forms = forms or s.forms
    years_back = years_back or s.years_back

    stats = IngestStats(errors=[])

    with EdgarClient() as client:
        tmap = client.ticker_map()

        for ticker in tickers:
            entry = tmap.get(ticker)
            if not entry:
                stats.errors.append(f"{ticker}: not found in EDGAR ticker map")
                continue
            cik = entry["cik"]
            upsert_company(cik, ticker, entry["title"])
            stats.companies += 1

            # ---- filings + sections
            try:
                subs = client.submissions(cik)
            except Exception as exc:
                stats.errors.append(f"{ticker}: submissions failed: {exc}")
                continue

            for row in recent_filings(subs, forms, limit=years_back):
                if progress:
                    progress(f"{ticker} {row['form']} {row['filing_date']}")
                url = client.filing_url(cik, row["accession"], row["primary_doc"])
                filing_id = upsert_filing(cik, row, url)
                if filing_id is None:
                    stats.skipped += 1
                    continue
                try:
                    html = client.filing_document(cik, row["accession"], row["primary_doc"])
                    sections = parse_filing(html)
                except Exception as exc:
                    stats.errors.append(f"{ticker} {row['accession']}: parse failed: {exc}")
                    continue
                stats.sections += replace_sections(filing_id, sections)
                stats.filings += 1

            # ---- XBRL ground truth
            try:
                cf = client.company_facts(cik)
                facts = extract_facts(cik, cf, forms=tuple(forms))
                stats.facts += upsert_facts(facts)
            except Exception as exc:
                stats.errors.append(f"{ticker}: companyfacts failed: {exc}")

    return stats


# ------------------------------------------------------------------ fixtures
def load_fixture(path: str) -> IngestStats:
    """Load a frozen mini-corpus.

    CI needs a corpus that never changes and never hits the network, otherwise
    the eval gate measures the SEC's uptime rather than your code. This loads
    exactly that.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    stats = IngestStats(errors=[])

    for comp in payload.get("companies", []):
        upsert_company(comp["cik"], comp.get("ticker"), comp["name"], comp.get("sic"))
        stats.companies += 1

    for filing in payload.get("filings", []):
        filing_id = upsert_filing(
            filing["cik"],
            {
                "accession": filing["accession"],
                "form": filing["form"],
                "filing_date": filing["filing_date"],
                "period_end": filing.get("period_end"),
                "primary_doc": filing.get("primary_doc"),
            },
            filing.get("source_url", ""),
        )
        if filing_id is None:
            continue
        stats.filings += 1
        rows = [
            (filing_id, sec.get("item"), sec.get("title"), i, len(sec["body"]), sec["body"])
            for i, sec in enumerate(filing.get("sections", []))
        ]
        db.execute("DELETE FROM sections WHERE filing_id = %s", (filing_id,))
        db.execute_many(
            """
            INSERT INTO sections (filing_id, item, title, ordinal, char_len, body)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
        stats.sections += len(rows)

    fact_rows = [
        (
            f["cik"],
            f.get("taxonomy", "us-gaap"),
            f["tag"],
            f.get("unit", "USD"),
            f["fiscal_year"],
            f.get("fiscal_period", "FY"),
            f.get("period_start"),
            f.get("period_end"),
            f["value"],
            f.get("accession"),
            f.get("form", "10-K"),
        )
        for f in payload.get("facts", [])
    ]
    if fact_rows:
        db.execute_many(
            """
            INSERT INTO xbrl_facts (cik, taxonomy, tag, unit, fiscal_year, fiscal_period,
                                    period_start, period_end, value, accession, form)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (cik, taxonomy, tag, unit, fiscal_year, fiscal_period, period_end)
            DO NOTHING
            """,
            fact_rows,
        )
        stats.facts = len(fact_rows)

    return stats
