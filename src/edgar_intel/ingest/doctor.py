"""Parser diagnostics against real filings.

The parser in `parse.py` works on the fixture corpus. It will not work
perfectly on real filings, because 10-K markup is genuinely inconsistent
between filers and between years — the section headings move, tables nest
inside tables, and some filers wrap everything in a structure the offsets
regex was not written for.

That is expected. The problem is that when it fails it fails *quietly*:
sections come back empty or fused, chunks lose their labels, retrieval gets
worse, and the whole thing looks like a model problem. You lose a day.

This module turns that silent failure into a report. It fetches one filing per
company, parses it, and grades the parse against checks that a correct parse
should pass — then tells you which filing to open and what to look at.

    edgar-intel ingest doctor
    edgar-intel ingest doctor --ticker NVDA --save-html

Nothing here writes to the database. It is safe to run repeatedly while you
iterate on the parser, which is the point.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import get_settings
from .edgar_client import EdgarClient, recent_filings
from .parse import find_item_offsets, html_to_text, split_sections

# Items a 10-K is required to contain. Missing one of these is the strongest
# signal that heading detection failed rather than that the filer omitted it.
EXPECTED_ITEMS = {"1", "1A", "7", "7A", "8"}

# A parsed MD&A section that contains no large numbers almost certainly lost
# its tables — the single most damaging parse failure for retrieval, and the
# one that is invisible without a check like this.
#
# One comma group, not two. Filings state amounts "in millions", so the common
# printed form is "383,285" rather than "383,285,000,000". Requiring two groups
# would report zero figures on a perfectly good parse — a false alarm that
# would send you debugging a parser that was working.
_BIG_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+|\d{4,}")
_TABLE_ROW = re.compile(r"\|")


@dataclass(slots=True)
class Check:
    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        return f"  [{'ok ' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass(slots=True)
class FilingDiagnosis:
    ticker: str
    accession: str
    fiscal_year: int | None
    html_bytes: int = 0
    text_chars: int = 0
    items_found: list[str] = field(default_factory=list)
    sections: int = 0
    checks: list[Check] = field(default_factory=list)
    error: str = ""

    @property
    def healthy(self) -> bool:
        return not self.error and all(c.passed for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["healthy"] = self.healthy
        return payload

    def render(self) -> str:
        head = f"{self.ticker} {self.accession} FY{self.fiscal_year or '?'}"
        if self.error:
            return f"{head}\n  [FAIL] fetch/parse raised: {self.error}"
        lines = [
            f"{head}  ({self.html_bytes // 1024}KB html -> "
            f"{self.text_chars // 1000}k chars, {self.sections} sections)"
        ]
        lines += [c.line() for c in self.checks]
        return "\n".join(lines)


def diagnose_html(ticker: str, accession: str, fiscal_year: int | None, html: str) -> FilingDiagnosis:
    """Run every check against one filing's HTML. No network, no database."""
    diag = FilingDiagnosis(
        ticker=ticker, accession=accession, fiscal_year=fiscal_year, html_bytes=len(html)
    )

    text = html_to_text(html)
    diag.text_chars = len(text)
    offsets = find_item_offsets(text)
    diag.items_found = [item for item, _ in offsets]
    sections = split_sections(text)
    diag.sections = len(sections)
    by_item = {s.item: s for s in sections if s.item}

    # 1. Did text extraction produce anything like a filing?
    diag.checks.append(
        Check(
            "text extracted",
            diag.text_chars > 20_000,
            f"{diag.text_chars:,} chars"
            + ("" if diag.text_chars > 20_000 else " — suspiciously short, markup may be unhandled"),
        )
    )

    # 2. Were the required Items located?
    missing = sorted(EXPECTED_ITEMS - set(diag.items_found))
    diag.checks.append(
        Check(
            "required items located",
            not missing,
            f"found {', '.join(diag.items_found) or 'none'}"
            + (f" — MISSING {', '.join(missing)}" if missing else ""),
        )
    )

    # 3. Did the offsets land on the body rather than the table of contents?
    #
    # Measured by how far apart the headings are, not by where the first one
    # sits. A table of contents lists every Item within a few hundred
    # characters, so a ToC match produces headings clustered together; a
    # correct parse produces headings spread across most of the document.
    #
    # An earlier version of this check flagged "first item appears early",
    # which fires on any short filing whose contents page is brief — a false
    # alarm on a correct parse.
    if len(offsets) >= 3:
        span = offsets[-1][1] - offsets[0][1]
        coverage = span / max(1, diag.text_chars)
        clustered = coverage < 0.20
        diag.checks.append(
            Check(
                "offsets on body, not table of contents",
                not clustered,
                f"{len(offsets)} headings spanning {coverage:.0%} of the document"
                + (" — clustered, so the ToC was matched" if clustered else ""),
            )
        )
    else:
        diag.checks.append(
            Check(
                "offsets on body, not table of contents",
                False,
                f"only {len(offsets)} headings found — too few to judge",
            )
        )

    # 4. Are sections plausibly sized?
    tiny = [s.item for s in sections if s.item and s.char_len < 500]
    diag.checks.append(
        Check(
            "sections non-trivial",
            not tiny,
            "all sections have body text"
            if not tiny
            else f"near-empty sections: {', '.join(tiny)} — boundaries likely wrong",
        )
    )

    # 5. Did MD&A keep its financial tables?
    mdna = by_item.get("7")
    if mdna:
        numbers = len(_BIG_NUMBER.findall(mdna.body))
        rows = len(_TABLE_ROW.findall(mdna.body))
        diag.checks.append(
            Check(
                "MD&A retained figures",
                numbers >= 20,
                f"{numbers} large numbers, {rows} table separators"
                + ("" if numbers >= 20 else " — tables were probably dropped"),
            )
        )
    else:
        diag.checks.append(Check("MD&A retained figures", False, "Item 7 not found at all"))

    # 6. Did Risk Factors stay prose rather than absorbing the financials?
    risk = by_item.get("1A")
    if risk:
        leaked = len(_BIG_NUMBER.findall(risk.body))
        diag.checks.append(
            Check(
                "risk factors not fused with financials",
                leaked < 200,
                f"{risk.char_len:,} chars, {leaked} large numbers"
                + (" — section boundary probably ran past Item 1A" if leaked >= 200 else ""),
            )
        )
    else:
        diag.checks.append(Check("risk factors not fused with financials", False, "Item 1A not found"))

    # 7. Are the item headings in document order?
    positions = [pos for _, pos in offsets]
    diag.checks.append(
        Check(
            "items in document order",
            positions == sorted(positions),
            "ascending" if positions == sorted(positions) else "OUT OF ORDER — check the regexes",
        )
    )

    return diag


def run(
    tickers: list[str] | None = None,
    form: str = "10-K",
    save_html_dir: str | None = None,
    progress=None,
) -> dict[str, Any]:
    """Fetch and diagnose the most recent filing for each ticker."""
    s = get_settings()
    tickers = [t.upper() for t in (tickers or s.universe)]
    diagnoses: list[FilingDiagnosis] = []

    with EdgarClient() as client:
        tmap = client.ticker_map()
        for ticker in tickers:
            entry = tmap.get(ticker)
            if not entry:
                diagnoses.append(
                    FilingDiagnosis(ticker, "-", None, error="not in EDGAR ticker map")
                )
                continue
            cik = entry["cik"]
            try:
                subs = client.submissions(cik)
                rows = recent_filings(subs, [form], limit=1)
                if not rows:
                    diagnoses.append(
                        FilingDiagnosis(ticker, "-", None, error=f"no recent {form}")
                    )
                    continue
                row = rows[0]
                if progress:
                    progress(f"{ticker} {row['accession']}")
                html = client.filing_document(cik, row["accession"], row["primary_doc"])

                if save_html_dir:
                    # Keeping the raw HTML is what makes the next debugging pass
                    # fast: you can re-run the parser against a saved filing
                    # without hitting EDGAR again.
                    os.makedirs(save_html_dir, exist_ok=True)
                    path = os.path.join(save_html_dir, f"{ticker}_{row['accession']}.html")
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(html)

                fy = int(str(row["period_end"])[:4]) if row.get("period_end") else None
                diagnoses.append(diagnose_html(ticker, row["accession"], fy, html))
            except Exception as exc:
                diagnoses.append(
                    FilingDiagnosis(ticker, "-", None, error=f"{type(exc).__name__}: {exc}")
                )

    healthy = [d for d in diagnoses if d.healthy]
    broken = [d for d in diagnoses if not d.healthy]

    return {
        "filings_checked": len(diagnoses),
        "healthy": len(healthy),
        "broken": len(broken),
        "diagnoses": [d.as_dict() for d in diagnoses],
        "verdict": _verdict(diagnoses),
    }


def _verdict(diagnoses: list[FilingDiagnosis]) -> str:
    broken = [d for d in diagnoses if not d.healthy]
    if not broken:
        return (
            "All filings parsed cleanly. Ingest with confidence — and note this in "
            "your notes, because it will stop being true when you add a filer whose "
            "markup differs."
        )

    # Which check fails most often tells you what to fix first.
    counts: dict[str, int] = {}
    for d in broken:
        for c in d.checks:
            if not c.passed:
                counts[c.name] = counts.get(c.name, 0) + 1
    worst = max(counts, key=counts.get) if counts else ""

    hints = {
        "required items located": (
            "ITEM_PATTERNS in ingest/parse.py does not match how these filers write "
            "their headings. Save the HTML with --save-html, grep it for 'Item 1A', "
            "and widen the regex to match what you find."
        ),
        "MD&A retained figures": (
            "Tables are being dropped during text extraction. Look at "
            "_FilingTextExtractor in ingest/parse.py — these filers probably nest "
            "tables or use a tag the cell handler ignores."
        ),
        "offsets on body, not table of contents": (
            "find_item_offsets matched the table of contents instead of the body. "
            "The 'last occurrence' rule is not enough for these filings; consider "
            "requiring headings to be spread across the document."
        ),
        "sections non-trivial": (
            "Section boundaries are collapsing. Check the offsets directly before "
            "blaming split_sections."
        ),
        "risk factors not fused with financials": (
            "A section boundary is running past where it should stop, so Item 1A is "
            "absorbing the financial statements. Usually the next Item's heading was "
            "not matched."
        ),
        "text extracted": (
            "Text extraction produced almost nothing. These filings may be a wrapper "
            "document pointing at the real filing, or use markup html.parser is "
            "mishandling."
        ),
        "items in document order": (
            "Headings matched out of order, which means a regex is matching body "
            "prose rather than a heading."
        ),
    }

    tickers = ", ".join(d.ticker for d in broken)
    return (
        f"{len(broken)} of {len(diagnoses)} filings failed: {tickers}. "
        f"Most common failure: '{worst}'. {hints.get(worst, 'Inspect the saved HTML.')}"
    )


def save_report(payload: dict[str, Any], path: str = "reports/parser_doctor.json") -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path
