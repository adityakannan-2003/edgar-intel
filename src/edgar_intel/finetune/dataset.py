"""Build a supervised fine-tuning dataset from filings + XBRL.

The task chosen for fine-tuning matters more than the training recipe, so here
is the reasoning.

Fine-tuning a small model to write good narrative analysis is a bad idea: the
output is open-ended, grading it needs a judge, and a frontier model will beat
you. Fine-tuning a small model to do *structured extraction* is a good idea:
the output is short and schema-constrained, correctness is checkable against
XBRL without a judge, and small models close most of the gap on narrow
extraction tasks at a fraction of the cost.

So the task is: given a passage from a filing, emit the financial facts stated
in it as JSON. Every pair is labelled from XBRL by distant supervision: a fact
is a label when the passage prints its value on a line that names it. Train
and held-out labels come from the same labeller, so the held-out score is
measured against labels with the labeller's own noise -- which is why a sample
of them is audited by hand before any number is quoted (`audit_sample`).

What the first version of this labeller got wrong, measured on 16 Oct 2026
over the 8-company corpus, because each one is an easy mistake to repeat:

- Substring matching. EPS 6.13 was searched as "6", which is in nearly every
  chunk: 18,480 of 19,199 labels were diluted EPS, and revenue had none.
- A cap of 8 facts per passage, filled by that EPS noise first.
- `LIMIT 400` chunks per company, ordered by id, which cut off Item 8 -- the
  financial statements, where most of the facts are.
- Any fiscal year back to 2007, so a 3-digit FY2009 figure matched a pension
  table cell in a FY2025 filing.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .. import db
from ..evals.goldenset import _canonicalize_fact_rows, _facts
from ..ingest.xbrl import CORE_TAGS
from ..retrieval.index import build_chunks
from .scoring import CONCEPTS, find_value, parse_extraction, score_extraction

__all__ = [
    "CONCEPTS",
    "FT_SECTORS",
    "INSTRUCTION",
    "Example",
    "add_negatives",
    "build_dataset",
    "build_examples",
    "parse_extraction",
    "score_extraction",
    "split",
]

INSTRUCTION = (
    "Extract the financial facts that a passage from an SEC filing states, for "
    "these concepts only: " + "; ".join(CONCEPTS) + ".\n\n"
    'Return a JSON array of {"concept": "<one of the concepts above>", '
    '"fiscal_year": <int>, "value": <number>}. Report only figures the passage '
    "itself prints, only for the fiscal years listed with it, and only company "
    "totals, not segments. fiscal_year is the calendar year the figure's fiscal "
    "year ends in (one ending in the first week of January counts as the year "
    "before), even where the company names its years differently. Give each "
    "value as a plain number, in full units or as printed. "
    "Return [] if the passage states none of these facts."
)

# A 10-K's income statement prints three years and its balance sheet two.
# Matching any year in XBRL's history (2007 onward) is what let 3-digit
# coincidences from FY2009 into the first version's labels.
YEARS_PRINTED = 3

# The fine-tune universe: ten sectors of four, so a company-disjoint split can
# hold one company out of every sector. It lives in its own database
# (`edgar_ft`), never in the RAG corpus: adding 32 companies there would add
# distractors to every retrieval run and break comparison with every report.
#
# Two first choices did not survive ingestion (16 Oct 2026). Banks (JPM, BAC,
# WFC, GS) file thousands of prospectuses a year, so EDGAR's "recent" index
# held one 10-K each, and they print their statements in an exhibit rather
# than the 10-K itself. XOM's ticker now maps to ExxonMobil Holdings Corp, a
# new registrant with no 10-K yet.
FT_SECTORS: dict[str, tuple[str, ...]] = {
    "hardware & semiconductors": ("AAPL", "NVDA", "INTC", "CSCO"),
    "software & internet": ("MSFT", "ORCL", "ADBE", "GOOGL"),
    "retail": ("COST", "WMT", "TGT", "HD"),
    "consumer staples": ("PG", "KO", "PEP", "CL"),
    "pharma": ("JNJ", "PFE", "MRK", "ABBV"),
    "health insurance & services": ("UNH", "CVS", "CI", "ELV"),
    "industrials": ("CAT", "DE", "HON", "LMT"),
    "energy": ("EOG", "CVX", "COP", "SLB"),
    "payments & business services": ("V", "MA", "ADP", "PYPL"),
    "telecom & media": ("VZ", "T", "CMCSA", "DIS"),
}

# A label must sit on a line that names its concept, or under a heading within
# three lines that does ("Earnings per share:" above a "Diluted" row). Without
# this, "Total liabilities and equity" is labelled as total assets -- the same
# number by the accounting identity, but not a statement of total assets.
CONCEPT_WORDS: dict[str, re.Pattern[str]] = {
    concept: re.compile(pattern, re.I)
    for concept, pattern in {
        "total revenue": r"revenue|net sales|\bsales\b",
        "net income": r"net\s+(?:\(loss\)\s+|\(losses\)\s+)?(?:income|earnings|loss)",
        "operating income": (
            r"operating\s+(?:\(loss\)\s+)?(?:income|profit|loss)"
            r"|(?:income|earnings|profit)\s+from\s+operations"
        ),
        "research and development expense": r"research,?\s+(?:and|&)\s+development|\br&d\b",
        "total assets": r"total\s+assets",
        "total liabilities": r"total\s+liabilities",
        "total stockholders' equity": r"equity",
        # Not bare "cash": "net cash used for investing activities was $5.5
        # billion" is a cash flow, and was labelled as PepsiCo's cash balance.
        "cash and cash equivalents": (
            r"cash\s+and\s+(?:cash\s+)?equivalents|cash\s+balances?|\bof\s+cash\b"
        ),
        "gross profit": r"gross\s+(?:profit|margin)",
        "diluted earnings per share": r"diluted|per\s+(?:common\s+)?share",
        "common shares outstanding": r"\bshares\b",
    }.items()
}
ANCHOR_LINES = 3

# A table row is short; a prose paragraph arrives as one long line, so "within
# three lines" would mean "anywhere in the paragraph". In prose the concept must
# be named in the same sentence, close to the number. The first audit found
# "capitalized software totaled $1.4 billion" labelled as cash, because "cash"
# appeared elsewhere in the paragraph.
PROSE_LINE_CHARS = 300
PROSE_BEFORE, PROSE_AFTER = 150, 40
_SENTENCE_BREAK = re.compile(r"\.\s|;\s|\n|•")

# Three comma-grouped figures make a passage a table for negative sampling;
# an exhibit number or "Item 601" does not.
TABLE_FIGURE = re.compile(r"(?<![\d,.])\d{1,3}(?:,\d{3})+(?![\d])")

# A figure worth stating: "1,234", "577", "$5.2 billion". Not a year: "In 2024,
# cash ..." is prose, and treating years as figures would make every dated
# sentence look like a table.
_BIG_NUMBER = re.compile(
    r"(?<![\d,.])\d{1,3}(?:,\d{3})+(?![\d])"
    r"|(?<![\d,.])(?!(?:19|20)\d{2}(?![\d,]))\d{3,}(?![\d,])"
    r"|\d+(?:\.\d+)?\s*(?:billion|million)",
    re.I,
)


@dataclass(slots=True)
class Example:
    id: str
    ticker: str
    company: str
    accession: str
    filing_fy: int
    period_end: str
    passage: str
    facts: list[dict[str, Any]]
    sector: str = ""
    # Labels the anchor rule dropped: the number is printed, the concept is
    # not named near it. Never trained on; the scorer ignores a prediction of
    # one rather than calling it right or wrong (`score_extraction(ignore=)`).
    dropped: list[dict[str, Any]] = field(default_factory=list)

    def years(self) -> list[int]:
        return list(range(self.filing_fy - YEARS_PRINTED + 1, self.filing_fy + 1))

    def user_message(self) -> str:
        """The header plus the passage. Every arm gets exactly this input.

        The header carries what the passage often lacks: whose filing, and
        which year's. A chunk that starts mid-table has no column headings, and
        without the filing year a model cannot place "18,315" in any year.
        """
        years = ", ".join(str(y) for y in self.years())
        return (
            f"Company: {self.company} ({self.ticker})\n"
            f"Filing: 10-K for fiscal year {self.filing_fy}, period ended {self.period_end}\n"
            f"Fiscal years to report: {years}\n\n"
            f"PASSAGE:\n{self.passage}"
        )

    def answer(self) -> str:
        return json.dumps(self.facts, separators=(",", ":"))

    def to_chat(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": self.user_message()},
                {"role": "assistant", "content": self.answer()},
            ]
        }

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(self.to_chat())
        return record


def _plain(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _header_end(passage: str) -> int:
    """Where the chunk's "[7 Management's Discussion ...]" prefix ends.

    The section title is not a statement the passage makes. In the first
    audit, "Equity" in "Market for Registrant's Common Equity" anchored a
    share-repurchase figure as stockholders' equity.
    """
    if passage.startswith("["):
        close = passage.find("]")
        if close > 0:
            return close + 1
    return 0


def _window(passage: str, start: int, end: int, prose: bool = False) -> tuple[int, int]:
    """The span a concept must be named in for the figure at [start, end).

    A table row: the row and the ANCHOR_LINES above it. Prose: the same
    sentence, at most PROSE_BEFORE characters before the figure and
    PROSE_AFTER after ("$16.56 per diluted share"). Never the section title.
    """
    floor = _header_end(passage)
    line_start = passage.rfind("\n", 0, start) + 1
    line_end = passage.find("\n", end)
    line_end = len(passage) if line_end < 0 else line_end

    if prose or line_end - line_start > PROSE_LINE_CHARS:
        lo = max(line_start, start - PROSE_BEFORE, floor)
        breaks = [m.end() for m in _SENTENCE_BREAK.finditer(passage, lo, start)]
        if breaks:
            lo = breaks[-1]
        hi = min(line_end, end + PROSE_AFTER)
        stop = _SENTENCE_BREAK.search(passage, end, hi)
        return lo, stop.start() if stop else hi

    lo = line_start
    for _ in range(ANCHOR_LINES):
        if lo == 0:
            break
        lo = passage.rfind("\n", 0, lo - 1) + 1
    return max(lo, floor), line_end


def _is_prose_match(match: re.Match[str]) -> bool:
    return "illion" in match.group(0).lower()


def _anchored(passage: str, match: re.Match[str], concept: str) -> bool:
    lo, hi = _window(passage, match.start(), match.end(), _is_prose_match(match))
    return bool(CONCEPT_WORDS[concept].search(passage[lo:hi]))


def candidate_facts(
    facts: list[dict[str, Any]], filing_fy: int
) -> dict[tuple[str, int], list[float]]:
    """Each (concept, year) this filing could print, with every value it might print.

    A year can carry two values: its own 10-K's figure, and the figure the
    next 10-K printed for it as a comparative (`prior_year_value`), which
    differs when the company recast it. Either one in the passage is a label.
    """
    years = set(range(filing_fy - YEARS_PRINTED + 1, filing_fy + 1))
    out: dict[tuple[str, int], list[float]] = {}
    for row in _canonicalize_fact_rows(facts):
        concept = CORE_TAGS.get(row["tag"])
        if concept is None:
            continue
        year = int(row["fiscal_year"])
        if year in years:
            out.setdefault((concept, year), []).append(float(row["value"]))
        prior = row.get("prior_year_value")
        if prior is not None and year - 1 in years and year <= filing_fy:
            values = out.setdefault((concept, year - 1), [])
            if float(prior) not in values:
                values.append(float(prior))
    return out


def label_passage(
    passage: str, candidates: dict[tuple[str, int], list[float]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """(labels, dropped, ambiguous) for one passage.

    `ambiguous` is set when two different facts share a printed number in this
    passage -- an unchanged share count, say -- so which one the passage
    states cannot be decided; the caller discards the passage.
    """
    hits: list[tuple[int, str, int, float, bool]] = []
    for (concept, year), values in candidates.items():
        for value in values:
            match = find_value(passage, value, concept)
            if match:
                hits.append((match.start(), concept, year, value, _anchored(passage, match, concept)))
                break

    starts = [h[0] for h in hits if h[4]]
    ambiguous = len(starts) != len(set(starts))

    hits.sort(key=lambda h: (h[0], -h[2]))
    labels, dropped = [], []
    for _start, concept, year, value, anchored in hits:
        fact = {"concept": concept, "fiscal_year": year, "value": _plain(value)}
        (labels if anchored else dropped).append(fact)
    return labels, dropped, ambiguous


def _filings(tickers: list[str] | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT f.id, f.accession, f.fiscal_year, f.period_end,
               co.cik, co.ticker, co.name
          FROM filings f
          JOIN companies co ON co.cik = f.cik
         WHERE f.form = '10-K' AND f.fiscal_year IS NOT NULL
    """
    params: list[Any] = []
    if tickers:
        sql += " AND co.ticker = ANY(%s)"
        params.append(tickers)
    sql += " ORDER BY co.ticker, f.fiscal_year"
    return db.query(sql, params)


def sector_of(ticker: str) -> str:
    for sector, tickers in FT_SECTORS.items():
        if ticker in tickers:
            return sector
    return "other"


def build_examples(
    tickers: list[str] | None = None,
    target_tokens: int = 512,
    overlap: int = 64,
    max_chars: int = 3000,
) -> tuple[list[Example], list[Example], dict[str, int]]:
    """(positives, negative candidates, counts), chunking every 10-K in memory.

    Chunks are rebuilt with the shipped section_aware settings rather than read
    from the `chunks` table, so the dataset does not depend on what has been
    indexed or embedded. A negative candidate has no label, no dropped label,
    and no line that both names a concept and prints a 3-digit-or-larger
    number -- a passage that might state a fact the labeller missed is never
    used to teach the model to say [].
    """
    positives: list[Example] = []
    negatives: list[Example] = []
    counts = {"filings": 0, "chunks": 0, "ambiguous": 0, "labels": 0, "dropped": 0}
    facts_by_cik: dict[str, list[dict[str, Any]]] = {}

    for filing in _filings(tickers):
        cik = filing["cik"]
        if cik not in facts_by_cik:
            facts_by_cik[cik] = _facts(cik)
        candidates = candidate_facts(facts_by_cik[cik], int(filing["fiscal_year"]))
        counts["filings"] += 1

        for chunk in build_chunks(filing["id"], "section_aware", target_tokens, overlap):
            counts["chunks"] += 1
            passage = chunk.body[:max_chars]
            labels, dropped, ambiguous = label_passage(passage, candidates)
            if ambiguous:
                counts["ambiguous"] += 1
                continue
            example = Example(
                id=f"{filing['accession']}:{chunk.ordinal}",
                ticker=filing["ticker"],
                company=filing["name"],
                accession=filing["accession"],
                filing_fy=int(filing["fiscal_year"]),
                period_end=str(filing["period_end"]),
                passage=passage,
                facts=labels,
                sector=sector_of(filing["ticker"]),
                dropped=dropped,
            )
            counts["labels"] += len(labels)
            counts["dropped"] += len(dropped)
            if labels:
                positives.append(example)
            elif not dropped and not _names_a_figure(passage):
                negatives.append(example)

    return positives, negatives, counts


def _names_a_figure(passage: str) -> bool:
    for line in passage.splitlines():
        if _BIG_NUMBER.search(line) and any(p.search(line) for p in CONCEPT_WORDS.values()):
            return True
    return False


def add_negatives(
    positives: list[Example], candidates: list[Example], ratio: float = 0.25, seed: int = 11
) -> list[Example]:
    """Add passages labelled [] at `ratio` per company, half of them numeric.

    Without negatives the model learns "always emit facts" and hallucinates
    numbers on prose. Half are drawn from tables -- three or more
    comma-grouped figures, a pension table or a debt schedule -- because
    prose-only negatives are too easy: a model can learn "no digits, no
    facts" and still invent revenue from a table. The rest have no figure.
    """
    rng = random.Random(seed)
    by_ticker: dict[str, list[Example]] = {}
    for ex in candidates:
        by_ticker.setdefault(ex.ticker, []).append(ex)

    chosen: list[Example] = []
    for ticker in sorted({e.ticker for e in positives}):
        want = round(ratio * sum(1 for e in positives if e.ticker == ticker))
        pool = by_ticker.get(ticker, [])
        numeric = [e for e in pool if len(TABLE_FIGURE.findall(e.passage)) >= 3]
        prose = [e for e in pool if not _BIG_NUMBER.search(e.passage)]
        rng.shuffle(numeric)
        rng.shuffle(prose)
        half = want // 2
        picked = numeric[:half] + prose[: want - half]
        if len(picked) < want:
            rest = [e for e in numeric[half:] + prose[want - half :] if e not in picked]
            picked += rest[: want - len(picked)]
        chosen += picked

    combined = positives + chosen
    rng.shuffle(combined)
    return combined


def split(
    examples: list[Example], held_per_sector: int = 1, seed: int = 13
) -> tuple[list[Example], list[Example]]:
    """Split by company, not by row, holding out companies from every sector.

    Splitting randomly would put chunks from the same filing on both sides and
    the held-out score would be inflated by memorised boilerplate. Grouping by
    company is the honest split for this data, and stratifying by sector keeps
    the held-out set from being, say, two tech companies.
    """
    rng = random.Random(seed)
    held: set[str] = set()
    tickers = sorted({e.ticker for e in examples})
    for sector in sorted({sector_of(t) for t in tickers}):
        members = [t for t in tickers if sector_of(t) == sector]
        if len(members) < 2:
            continue
        rng.shuffle(members)
        held.update(members[:held_per_sector])
    return (
        [e for e in examples if e.ticker not in held],
        [e for e in examples if e.ticker in held],
    )


def write_jsonl(examples: list[Example], path: str) -> int:
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_record()) + "\n")
    return len(examples)


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def build_dataset(
    out_dir: str = "data/finetune",
    negatives_ratio: float = 0.25,
    tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Build, split and write the dataset, plus a manifest that freezes it.

    `split.json` records the companies on each side and the sha256 of both
    files. It is written before any model sees the data, and every later
    report quotes the held-out hash, so a score can always be tied to the
    exact examples it was measured on.
    """
    from ..evals.runner import git_sha

    os.makedirs(out_dir, exist_ok=True)
    positives, candidates, counts = build_examples(tickers)
    examples = add_negatives(positives, candidates, negatives_ratio)
    train, held_out = split(examples)

    train_path = os.path.join(out_dir, "train.jsonl")
    held_path = os.path.join(out_dir, "held_out.jsonl")
    write_jsonl(train, train_path)
    write_jsonl(held_out, held_path)

    def side(rows: list[Example]) -> dict[str, Any]:
        return {
            "examples": len(rows),
            "positives": sum(1 for e in rows if e.facts),
            "negatives": sum(1 for e in rows if not e.facts),
            "facts": sum(len(e.facts) for e in rows),
            "companies": sorted({e.ticker for e in rows}),
            "accessions": sorted({e.accession for e in rows}),
        }

    manifest = {
        "git_sha": git_sha(),
        "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        "negatives_ratio": negatives_ratio,
        "years_printed": YEARS_PRINTED,
        "counts": counts,
        "negative_candidates": len(candidates),
        "train": {**side(train), "sha256": sha256_file(train_path)},
        "held_out": {**side(held_out), "sha256": sha256_file(held_path)},
        "sectors": {
            sector: {
                "train": sorted({e.ticker for e in train if e.sector == sector}),
                "held_out": sorted({e.ticker for e in held_out if e.sector == sector}),
            }
            for sector in sorted({e.sector for e in examples})
        },
    }
    with open(os.path.join(out_dir, "split.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


# -------------------------------------------------------------------- audit
def load_records(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _context(passage: str, match: re.Match[str]) -> str:
    """The anchor window around a match, with the match marked »like this«."""
    start, end = match.start(), match.end()
    lo, hi = _window(passage, start, end, _is_prose_match(match))
    return passage[lo:start] + "»" + passage[start:end] + "«" + passage[end:hi]


def audit_sample(
    held_out_path: str, n_labels: int = 50, n_negatives: int = 10, seed: int = 29
) -> dict[str, Any]:
    """A seeded random sample of held-out labels (and negatives) to check by hand.

    Labels are sampled uniformly over facts, not passages, so the measured
    share that are right is the label precision the held-out F1 is scored
    against. Each item shows the line the labeller matched and the lines above
    it; `verdict` is filled in by a person: "right", "wrong" or "unsure".
    """
    rng = random.Random(seed)
    rows = load_records(held_out_path)
    facts = [(r, f) for r in rows for f in r["facts"]]
    items = []
    for i, (row, f) in enumerate(rng.sample(facts, min(n_labels, len(facts))), 1):
        match = find_value(row["passage"], f["value"], f["concept"])
        context = _context(row["passage"], match) if match else ""
        items.append({
            "n": i, "example_id": row["id"], "ticker": row["ticker"],
            "filing_fy": row["filing_fy"], "concept": f["concept"],
            "fiscal_year": f["fiscal_year"], "value": f["value"],
            "context": context, "verdict": None, "note": "",
        })
    negatives = [r for r in rows if not r["facts"]]
    neg_items = [
        {"n": i, "example_id": r["id"], "ticker": r["ticker"], "filing_fy": r["filing_fy"],
         "passage": r["passage"], "verdict": None, "note": ""}
        for i, r in enumerate(rng.sample(negatives, min(n_negatives, len(negatives))), 1)
    ]
    return {
        "held_out_path": held_out_path,
        "held_out_sha256": sha256_file(held_out_path),
        "seed": seed,
        "labels": items,
        "negatives": neg_items,
    }
