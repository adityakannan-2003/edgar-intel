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
in it as JSON. Every training pair is labelled from XBRL, so the labels are
externally verified rather than model-generated -- which also means the
benchmark comparing this against prompting is measuring something real.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any

from .. import db
from ..ingest.xbrl import CORE_TAGS

INSTRUCTION = (
    "Extract every financial fact stated in the passage. Return a JSON array of "
    '{"tag": "<XBRL tag>", "fiscal_year": <int>, "value": <number>, "unit": "<unit>"}. '
    "Return [] if the passage states no financial facts."
)


@dataclass(slots=True)
class Example:
    passage: str
    facts: list[dict[str, Any]]
    ticker: str
    fiscal_year: int

    def to_chat(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": self.passage},
                {"role": "assistant", "content": json.dumps(self.facts, separators=(",", ":"))},
            ]
        }


def _number_variants(value: float) -> list[str]:
    """How a filing might print this figure.

    Filings state amounts "in millions" or "in thousands" in a table header and
    then print the scaled number. Matching only the raw value would find almost
    nothing, which would silently produce an empty training set.
    """
    out: list[str] = []
    for scale in (1, 1_000, 1_000_000):
        scaled = abs(value) / scale
        if scaled < 1:
            continue
        out.append(f"{scaled:,.0f}")
        if scaled < 1000:
            out.append(f"{scaled:,.2f}")
    return list(dict.fromkeys(out))


def build_examples(
    strategy: str = "section_aware",
    max_per_company: int = 400,
    min_facts: int = 1,
) -> list[Example]:
    """Label chunks by finding XBRL values that literally appear in them.

    Distant supervision, with the usual caveat: a chunk containing the string
    "383,285" is probably stating total net sales, but might be stating
    something else that happens to equal it. That noise is real and worth
    naming rather than hiding -- it is also why the eval set is generated
    independently from XBRL rather than from these labels.
    """
    examples: list[Example] = []

    companies = db.query("SELECT cik, ticker FROM companies ORDER BY ticker")
    for comp in companies:
        rows = db.query(
            """
            SELECT c.id, c.body, f.fiscal_year
              FROM chunks c
              JOIN filings f ON f.id = c.filing_id
             WHERE c.strategy = %s AND f.cik = %s
             ORDER BY c.id
             LIMIT %s
            """,
            (strategy, comp["cik"], max_per_company),
        )
        facts = db.query(
            """
            SELECT tag, unit, fiscal_year, value
              FROM xbrl_facts
             WHERE cik = %s AND fiscal_period = 'FY' AND tag = ANY(%s)
            """,
            (comp["cik"], list(CORE_TAGS)),
        )
        if not rows or not facts:
            continue

        for row in rows:
            body = row["body"]
            body_norm = body.replace(",", "")
            found: list[dict[str, Any]] = []
            for fact in facts:
                value = float(fact["value"])
                for variant in _number_variants(value):
                    if variant in body or variant.replace(",", "") in body_norm:
                        found.append(
                            {
                                "tag": fact["tag"],
                                "fiscal_year": int(fact["fiscal_year"]),
                                "value": value,
                                "unit": fact["unit"],
                            }
                        )
                        break
            if len(found) >= min_facts:
                examples.append(
                    Example(
                        passage=body[:3000],
                        facts=found[:8],
                        ticker=comp["ticker"] or comp["cik"],
                        fiscal_year=int(row["fiscal_year"] or 0),
                    )
                )

    return examples


def add_negatives(examples: list[Example], strategy: str, ratio: float = 0.25,
                  seed: int = 11) -> list[Example]:
    """Add passages with no financial facts, labelled with an empty array.

    Without negatives the model learns "always emit facts" and hallucinates
    numbers on prose. This is the single most common omission in extraction
    fine-tunes and it shows up immediately as precision collapse.
    """
    rng = random.Random(seed)
    want = int(len(examples) * ratio)
    if want <= 0:
        return examples

    rows = db.query(
        """
        SELECT c.body, f.fiscal_year, co.ticker
          FROM chunks c
          JOIN filings f ON f.id = c.filing_id
          JOIN companies co ON co.cik = f.cik
         WHERE c.strategy = %s
           AND c.body !~ '[0-9]{3},[0-9]{3}'
         ORDER BY random()
         LIMIT %s
        """,
        (strategy, want),
    )
    negatives = [
        Example(passage=r["body"][:3000], facts=[], ticker=r["ticker"] or "?",
                fiscal_year=int(r["fiscal_year"] or 0))
        for r in rows
    ]
    combined = examples + negatives
    rng.shuffle(combined)
    return combined


def split(
    examples: list[Example], train: float = 0.8, seed: int = 13
) -> tuple[list[Example], list[Example]]:
    """Split by company, not by row.

    Splitting randomly would put chunks from the same filing on both sides and
    the held-out score would be inflated by memorised boilerplate. Grouping by
    company is the honest split for this data.
    """
    rng = random.Random(seed)
    tickers = sorted({e.ticker for e in examples})
    rng.shuffle(tickers)
    cut = max(1, int(len(tickers) * train))
    train_tickers = set(tickers[:cut])
    return (
        [e for e in examples if e.ticker in train_tickers],
        [e for e in examples if e.ticker not in train_tickers],
    )


def write_jsonl(examples: list[Example], path: str) -> int:
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_chat()) + "\n")
    return len(examples)


def build_dataset(
    out_dir: str = "data/finetune",
    strategy: str = "section_aware",
    negatives_ratio: float = 0.25,
) -> dict[str, Any]:
    import os

    os.makedirs(out_dir, exist_ok=True)
    examples = build_examples(strategy=strategy)
    examples = add_negatives(examples, strategy, negatives_ratio)
    train, held_out = split(examples)

    n_train = write_jsonl(train, os.path.join(out_dir, "train.jsonl"))
    n_eval = write_jsonl(held_out, os.path.join(out_dir, "held_out.jsonl"))

    return {
        "train": n_train,
        "held_out": n_eval,
        "total": len(examples),
        "negatives": sum(1 for e in examples if not e.facts),
        "train_companies": sorted({e.ticker for e in train}),
        "held_out_companies": sorted({e.ticker for e in held_out}),
    }


EXTRACT_JSON = re.compile(r"\[.*\]", re.S)


def parse_extraction(text: str) -> list[dict[str, Any]]:
    """Parse a model's extraction output, tolerating surrounding prose."""
    match = EXTRACT_JSON.search(text)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def score_extraction(
    predicted: list[dict[str, Any]], expected: list[dict[str, Any]], tol: float = 0.005
) -> dict[str, float]:
    """Set-level precision/recall/F1 over (tag, fiscal_year, value) triples.

    Values match within a relative tolerance, so a rounding difference is not
    scored as a miss.
    """
    def key(fact: dict[str, Any]) -> tuple:
        return (str(fact.get("tag")), int(fact.get("fiscal_year") or 0))

    exp_by_key: dict[tuple, float] = {
        key(f): float(f.get("value") or 0) for f in expected
    }
    pred_by_key: dict[tuple, float] = {
        key(f): float(f.get("value") or 0) for f in predicted if isinstance(f, dict)
    }

    tp = 0
    for k, pred_value in pred_by_key.items():
        if k not in exp_by_key:
            continue
        exp_value = exp_by_key[k]
        if exp_value == 0:
            tp += int(pred_value == 0)
        elif abs(pred_value - exp_value) / abs(exp_value) <= tol:
            tp += 1

    precision = tp / len(pred_by_key) if pred_by_key else (1.0 if not exp_by_key else 0.0)
    recall = tp / len(exp_by_key) if exp_by_key else (1.0 if not pred_by_key else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_set_match": float(
            tp == len(exp_by_key) == len(pred_by_key)
        ),
    }
