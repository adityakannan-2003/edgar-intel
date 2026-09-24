"""Surfacing the filing text a narrative reference answer has to be written from.

Expanding the narrative set from 24 to 40 cases means writing 16 new reference
answers. A reference answer is ground truth: if it is written from memory of
what these companies probably disclose, the eval set stops measuring the system
and starts measuring the recollection of whoever wrote it. Every one of the
existing 24 was written against the filing section in
`reports/narrative_source_sections.txt`; this module is the tool that produced
that file, rebuilt so the next 16 are dumped the same way instead of by a
one-off heredoc that was not kept.

Two things it deliberately does not do:

  It does not require a reference to exist. `generate_narrative_cases` raises on
  a missing reference, which is right for the golden set and useless here --
  the whole point is to dump the sections for cases that have no reference yet.
  So the company x seed expansion lives in `narrative_plan`, which needs no
  references, and `generate_narrative_cases` consumes it and enforces them.

  It does not quietly hand back nothing when the expected Item is absent.
  Incorporation by reference is routine -- NVIDIA's Item 3 points at a note,
  P&G's Item 7A at the MD&A -- so a case whose expected Item has no section is
  reported as such, with the Items the filing *does* have and their sizes, and
  the reference answer then has to be written from wherever the disclosure
  actually lives. Silence here would look identical to "the company discloses
  nothing", which is the sort of confusion that put a fiscal year from the
  report onto a fact from a different period.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import db

SEPARATOR = "=" * 100


@dataclass
class NarrativePlanItem:
    """One (company, topic) cell of the narrative grid, before references exist."""

    case_id: str
    cik: str
    ticker: str | None
    company: str
    slug: str
    item: str
    question: str
    fiscal_year: int | None
    has_reference: bool


def narrative_plan(max_per_company: int = 5, seed: int = 7) -> list[NarrativePlanItem]:
    """The (company, topic) cells `generate_narrative_cases` would produce.

    The seeded shuffle is reproduced exactly -- one `Random(seed)` advanced
    across companies, not re-seeded per company -- so `max_per_company=3` yields
    the same 24 case ids that are already in the golden set. That is asserted by
    a test: if the ordering ever drifts, the cases already labelled by hand stop
    matching the cases the generator emits, and the labels silently become
    labels of something else.
    """
    from .goldenset import (
        NARRATIVE_QUESTION_OVERRIDES,
        NARRATIVE_REFERENCES,
        NARRATIVE_SEEDS,
    )

    rng = random.Random(seed)
    plan: list[NarrativePlanItem] = []
    for row in db.query("SELECT cik, ticker, name FROM companies ORDER BY ticker"):
        seeds = NARRATIVE_SEEDS[:]
        rng.shuffle(seeds)
        for seed_case in seeds[:max_per_company]:
            year_row = db.query_one(
                "SELECT MAX(fiscal_year) AS fy FROM filings WHERE cik = %s", (row["cik"],)
            )
            fy = year_row["fy"] if year_row else None
            case_id = f"nar-{row['ticker'] or row['cik']}-{seed_case['slug']}"
            plan.append(
                NarrativePlanItem(
                    case_id=case_id,
                    cik=row["cik"],
                    ticker=row["ticker"],
                    company=row["name"],
                    slug=seed_case["slug"],
                    item=seed_case["item"],
                    question=NARRATIVE_QUESTION_OVERRIDES.get(
                        case_id, seed_case["question"].format(company=row["name"])
                    ),
                    fiscal_year=fy,
                    has_reference=case_id in NARRATIVE_REFERENCES,
                )
            )
    return plan


def missing_references(plan: list[NarrativePlanItem]) -> list[NarrativePlanItem]:
    return [item for item in plan if not item.has_reference]


# ------------------------------------------------------------------ sections
@dataclass
class SectionText:
    item: str | None
    title: str | None
    ordinal: int
    char_len: int
    body: str


@dataclass
class CaseSources:
    plan: NarrativePlanItem
    accession: str | None
    period_end: Any
    sections: list[SectionText] = field(default_factory=list)
    available_items: list[tuple[str | None, int]] = field(default_factory=list)
    filing_fiscal_year: int | None = None

    @property
    def year_mismatch(self) -> bool:
        """The case's year and the dumped filing's year disagree.

        The case year is `MAX(fiscal_year)` over *all* of a company's filings,
        including 10-Qs; the sections come from the latest 10-K. A newer 10-Q can
        make those differ, and then a reference answer written from FY2025 text
        would be labelled FY2026. Same shape as the fiscal-year bug: a year taken
        from one record attached to content from another.
        """
        return (
            self.filing_fiscal_year is not None
            and self.plan.fiscal_year is not None
            and int(self.filing_fiscal_year) != int(self.plan.fiscal_year)
        )

    @property
    def expected_item_found(self) -> bool:
        return bool(self.sections)

    @property
    def total_chars(self) -> int:
        return sum(s.char_len for s in self.sections)


def _latest_filing(cik: str) -> dict[str, Any] | None:
    """The filing the case's fiscal year refers to.

    Ordered by fiscal_year then filing_date so an amended or later-filed
    document for the same year wins, matching how the golden set picks the year
    (`MAX(fiscal_year)`) rather than picking by filing date alone.
    """
    return db.query_one(
        """
        SELECT id, accession, fiscal_year, period_end
          FROM filings
         WHERE cik = %s AND form = '10-K' AND fiscal_year IS NOT NULL
         ORDER BY fiscal_year DESC, filing_date DESC
         LIMIT 1
        """,
        (cik,),
    )


def collect_sources(item: NarrativePlanItem) -> CaseSources:
    filing = _latest_filing(item.cik)
    if filing is None:
        return CaseSources(plan=item, accession=None, period_end=None)

    rows = db.query(
        """
        SELECT item, title, ordinal, char_len, body
          FROM sections
         WHERE filing_id = %s
         ORDER BY ordinal
        """,
        (filing["id"],),
    )
    wanted = [
        SectionText(r["item"], r["title"], r["ordinal"], r["char_len"], r["body"])
        for r in rows
        if (r["item"] or "").strip().upper() == item.item.strip().upper()
    ]
    return CaseSources(
        plan=item,
        accession=filing["accession"],
        period_end=filing["period_end"],
        sections=wanted,
        available_items=[(r["item"], r["char_len"]) for r in rows],
        filing_fiscal_year=filing["fiscal_year"],
    )


# ---------------------------------------------------------------------- dump
def default_dump_name(
    n_cases: int, label: str = "", prefix: str = "reports/narrative_source_sections"
) -> str:
    """A name that encodes what was dumped, for the same reason the probe does.

    Two context-probe arms were both written to `context_probe_16k.json` and one
    run's evidence was lost. A sources dump is worse to lose: the reference
    answers written from it are the eval set's ground truth, and a file whose
    name does not say which cases it holds cannot be used to check them later.
    """
    parts = [f"{n_cases}cases"]
    if label:
        parts.append(label.replace(" ", "-"))
    return f"{prefix}_{'_'.join(parts)}.txt"


def render(sources: CaseSources, max_section_chars: int = 0) -> str:
    """One case block, in the format of the existing 24-case dump."""
    plan = sources.plan
    lines = [
        SEPARATOR,
        f"CASE ID: {plan.case_id}",
        f"QUESTION: {plan.question}",
        f"TICKER: {plan.ticker or plan.cik}",
        f"FISCAL YEAR: {plan.fiscal_year}",
        f"EXPECTED ITEM: {plan.item}",
        f"HAS REFERENCE: {'yes' if plan.has_reference else 'NO -- to be written'}",
        f"ACCESSION: {sources.accession or 'none'}",
        f"SECTIONS FROM 10-K FY: {sources.filing_fiscal_year}",
    ]
    if sources.year_mismatch:
        lines.append(
            f"WARNING: case year is FY{plan.fiscal_year} but these sections come "
            f"from the FY{sources.filing_fiscal_year} 10-K. Write the reference "
            f"from the text below and treat FY{sources.filing_fiscal_year} as its year."
        )

    if not sources.expected_item_found:
        available = ", ".join(
            f"{it or '?'} ({n:,}c)" for it, n in sources.available_items
        )
        lines += [
            "",
            f"NO SECTION FOR ITEM {plan.item}. The disclosure is either absent or "
            "incorporated by reference; write the reference from wherever it lives.",
            f"ITEMS PRESENT: {available or 'none -- filing has no parsed sections'}",
            "",
        ]
        return "\n".join(lines)

    for n, section in enumerate(sources.sections, start=1):
        body = section.body
        note = ""
        if max_section_chars and len(body) > max_section_chars:
            body = body[:max_section_chars]
            note = (
                f"\n[TRUNCATED: showing {max_section_chars:,} of "
                f"{section.char_len:,} chars]"
            )
        lines += [
            "",
            f"--- SECTION {n} ---",
            f"ITEM: {section.item}",
            f"TITLE: {section.title}",
            f"CHARS: {section.char_len:,}",
            body + note,
        ]
    lines.append("")
    return "\n".join(lines)


def dump(
    plan: list[NarrativePlanItem],
    out_path: str,
    overwrite: bool = False,
    max_section_chars: int = 0,
    progress: Callable[[CaseSources], None] | None = None,
) -> dict[str, Any]:
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f"{out_path} exists. Pass --overwrite, or choose another name -- "
            "a sources dump is the provenance of the reference answers written "
            "from it."
        )

    blocks: list[str] = []
    rows: list[dict[str, Any]] = []
    for item in plan:
        sources = collect_sources(item)
        blocks.append(render(sources, max_section_chars=max_section_chars))
        rows.append(
            {
                "case_id": item.case_id,
                "item": item.item,
                "fiscal_year": item.fiscal_year,
                "sections": len(sources.sections),
                "chars": sources.total_chars,
                "reference": "yes" if item.has_reference else "MISSING",
                "status": (
                    "NO SECTION"
                    if not sources.expected_item_found
                    else ("YEAR MISMATCH" if sources.year_mismatch else "ok")
                ),
            }
        )
        if progress:
            progress(sources)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(blocks))

    return {
        "out_path": out_path,
        "cases": len(plan),
        "missing_sections": [r["case_id"] for r in rows if r["status"] == "NO SECTION"],
        "year_mismatches": [r["case_id"] for r in rows if r["status"] == "YEAR MISMATCH"],
        "total_chars": sum(r["chars"] for r in rows),
        "rows": rows,
    }
