"""Parse and score extraction outputs, for every arm alike.

The benchmark's claim rests on this file. The prompted arms and the fine-tuned
arm are parsed by `parse_extraction` and scored by `score_extraction`, so a
difference between arms is a difference between models, not between graders.

Three decisions here change the numbers, so they are stated rather than buried:

1. Facts are keyed on the *concept* ("total revenue"), not the XBRL tag. Apple
   reports revenue as `RevenueFromContractWithCustomerExcludingAssessedTax`,
   UnitedHealth as `Revenues`. A model reading an unseen company's passage
   cannot know which, and scoring on the tag would mark a right answer wrong.
   A raw tag is still accepted and mapped through `CORE_TAGS`.
2. A value matches at any common reporting scale, the rule `grade_numeric`
   uses for the eval set. Many passages print "383,285" with the "(in
   millions)" header in another chunk, so the scale is often not in the input.
3. Values compare on magnitude. "Net loss | (18,756)" and "net income -18,756"
   state the same fact; a sign disagreement is counted in `diagnose`, not
   scored as a miss.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..ingest.xbrl import CORE_TAGS

# The answer vocabulary: CORE_TAGS' labels, in order, without duplicates. Two
# revenue tags and two R&D tags collapse to one concept each.
CONCEPTS: tuple[str, ...] = tuple(dict.fromkeys(CORE_TAGS.values()))
EPS_CONCEPT = "diluted earnings per share"

# The grader's reporting scales, plus billions: prose prints "$391.0 billion".
SCALES = (1, 1_000, 1_000_000, 1_000_000_000)

_TAG_TO_CONCEPT = {tag.lower(): label for tag, label in CORE_TAGS.items()}


def concept_of(name: Any) -> str | None:
    """The concept a model's `concept` (or `tag`) field names, or None."""
    if not isinstance(name, str):
        return None
    key = name.strip().replace("’", "'").lower()
    if key in CONCEPTS:
        return key
    return _TAG_TO_CONCEPT.get(key)


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace("$", "").replace(",", "").replace(" ", "")
        negative = text.startswith("(") and text.endswith(")")
        text = text.strip("()")
        try:
            number = float(text)
        except ValueError:
            return None
        return -number if negative else number
    return None


def _to_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        match = re.search(r"(?:19|20)\d{2}", value)
        return int(match.group(0)) if match else None
    return None


def normalise_fact(fact: Any) -> tuple[str, int, float] | None:
    """(concept, fiscal_year, value) for one emitted fact, or None if unusable."""
    if not isinstance(fact, dict):
        return None
    concept = concept_of(fact.get("concept", fact.get("tag")))
    year = _to_year(fact.get("fiscal_year"))
    value = _to_number(fact.get("value"))
    if concept is None or year is None or value is None:
        return None
    return concept, year, value


def values_match(predicted: float, expected: float, tol: float = 0.005) -> bool:
    """Whether `predicted`, read at some reporting scale, is `expected`."""
    p, e = abs(predicted), abs(expected)
    if e == 0:
        return p == 0
    return any(abs(p * scale - e) / e <= tol for scale in SCALES)


# ------------------------------------------------------------------ parsing
_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_extraction(text: str) -> list[dict[str, Any]]:
    """Parse a model's output into a list of fact dicts, tolerating wrappers.

    Accepts a bare array, an array inside a code fence or prose, or an object
    holding one (`{"facts": [...]}`). Reasoning models' `<think>` blocks are
    dropped first, because the brackets inside them are not the answer. Returns
    [] when nothing parses; `parse_ok` tells that apart from a real [].
    """
    return _parse(text)[0]


def parse_ok(text: str) -> bool:
    """Whether the output contained a JSON array at all."""
    return _parse(text)[1]


def _parse(text: str) -> tuple[list[dict[str, Any]], bool]:
    text = _THINK.sub("", text or "")
    decoder = json.JSONDecoder()
    for start in [i for i, ch in enumerate(text) if ch in "[{"]:
        try:
            data, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list)]
            if len(lists) != 1:
                continue
            data = lists[0]
        # "[1]" in prose is a footnote, not the answer: keep scanning past a
        # list that has items and no fact-shaped one.
        if isinstance(data, list) and (not data or any(isinstance(d, dict) for d in data)):
            return [d for d in data if isinstance(d, dict)], True
    return [], False


# ------------------------------------------------------------------ scoring
def _keyed(facts: list[dict[str, Any]]) -> tuple[dict[tuple[str, int], float], int]:
    """First value per (concept, year), and how many facts were unusable."""
    keyed: dict[tuple[str, int], float] = {}
    invalid = 0
    for fact in facts:
        norm = normalise_fact(fact)
        if norm is None:
            invalid += 1
            continue
        concept, year, value = norm
        keyed.setdefault((concept, year), value)
    return keyed, invalid


def _drop_ignored(
    pred: dict[tuple[str, int], float],
    ignore: list[dict[str, Any]] | None,
    tol: float,
    verifiable: list[list[Any]] | None = None,
) -> tuple[dict[tuple[str, int], float], int, int]:
    """(kept, ignored, unverifiable): predictions left to score, and why the rest went."""
    skip, _ = _keyed(ignore or [])
    kept = {
        k: v for k, v in pred.items() if not (k in skip and values_match(v, skip[k], tol))
    }
    ignored = len(pred) - len(kept)
    if verifiable is None:
        return kept, ignored, 0
    checkable = {(c, int(y)) for c, y in verifiable}
    scored = {k: v for k, v in kept.items() if k in checkable}
    return scored, ignored, len(kept) - len(scored)


def score_extraction(
    predicted: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    tol: float = 0.005,
    ignore: list[dict[str, Any]] | None = None,
    verifiable: list[list[Any]] | None = None,
) -> dict[str, float]:
    """Set-level precision/recall/F1 over (concept, fiscal_year, value) triples.

    An unusable emitted fact (unknown concept, no year, no number) counts as a
    false positive: the model said something and it cannot be right. `tp`,
    `n_pred` and `n_expected` are returned so the benchmark can report micro F1
    across passages, not only the per-passage mean.

    `ignore` holds facts the passage prints without naming them nearby -- the
    labeller's dropped labels. Predicting one is neither credited nor
    penalised, for every arm alike: "BALANCE AT SEPTEMBER 3, 2023 | ... |
    25,058" is total equity only by its column heading, which the passage may
    not contain.

    `verifiable`, when given, lists the (concept, year) pairs XBRL has a fact
    for; a prediction of any other pair is set aside as unverifiable, again
    for every arm alike.
    """
    exp, _ = _keyed(expected)
    pred, invalid = _keyed(predicted)
    pred, ignored, unverifiable = _drop_ignored(pred, ignore, tol, verifiable)

    tp = sum(
        1 for key, value in pred.items() if key in exp and values_match(value, exp[key], tol)
    )
    n_pred = len(pred) + invalid
    n_exp = len(exp)

    precision = tp / n_pred if n_pred else (1.0 if not n_exp else 0.0)
    recall = tp / n_exp if n_exp else (1.0 if not n_pred else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_set_match": float(tp == n_exp == n_pred),
        "tp": tp,
        "n_pred": n_pred,
        "n_expected": n_exp,
        "invalid": invalid,
        "ignored": ignored,
        "unverifiable": unverifiable,
    }


# ------------------------------------------------------------- diagnostics
_LEFT = r"(?<![\d,.])"
_RIGHT = r"(?![\d]|[.,]\d|\s*%)"


def printed_forms(value: float, concept: str) -> list[str]:
    """How a filing's table prints this figure, most specific first.

    A table "in millions" prints XBRL's 383,285,000,000 as "383,285". The value
    is an exact multiple of its reporting scale, so a scale it is not an exact
    multiple of is skipped -- which is what keeps "606" from being generated
    for 606,200,000 shares. Figures under 100 at their scale are never matched
    as bare numbers: a two-digit cell coincides with something on every page.
    EPS is printed to the cent.
    """
    v = abs(value)
    if concept == EPS_CONCEPT:
        return [f"{v:.2f}"] if v >= 0.01 else []
    forms = []
    for scale in (1, 1_000, 1_000_000):
        scaled = v / scale
        if scaled >= 100 and abs(scaled - round(scaled)) < 1e-6:
            forms.append(f"{round(scaled):,}")
    return forms


def prose_forms(value: float, concept: str) -> list[str]:
    """How MD&A prose prints it: "$391.0 billion", "$577.5 million"."""
    v = abs(value)
    if concept in (EPS_CONCEPT, "common shares outstanding"):
        return []
    forms = []
    if v >= 1e9:
        forms += [f"{v / 1e9:.1f} billion", f"{v / 1e9:.2f} billion"]
    if 1e7 <= v < 1e10:
        forms.append(f"{v / 1e6:.1f} million")
    return forms


def find_value(passage: str, value: float, concept: str) -> re.Match[str] | None:
    """The first place `passage` prints `value`, or None."""
    for form in printed_forms(value, concept):
        match = re.search(_LEFT + re.escape(form) + _RIGHT, passage)
        if match:
            return match
    for form in prose_forms(value, concept):
        number, word = form.split(" ")
        match = re.search(_LEFT + re.escape(number) + r"\s*" + word, passage)
        if match:
            return match
    return None


def value_in_passage(passage: str, value: float) -> bool:
    """Whether the passage prints this number at any scale (for diagnosis)."""
    for scale in SCALES:
        scaled = abs(value) / scale
        if scaled < 1:
            break
        forms = {f"{scaled:,.2f}", f"{scaled:,.1f}"}
        # Rounded to a whole number only when that is not most of the figure:
        # "8" must not count as printing 8.28.
        if scaled >= 100 or float(scaled).is_integer():
            forms.add(f"{scaled:,.0f}")
        for form in forms:
            if re.search(_LEFT + re.escape(form) + _RIGHT, passage):
                return True
    return False


def diagnose(
    predicted: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    passage: str,
    tol: float = 0.005,
    ignore: list[dict[str, Any]] | None = None,
    verifiable: list[list[Any]] | None = None,
) -> dict[str, int]:
    """Count each error by kind, deterministically. This is "where it breaks".

    For a predicted fact that is not a true positive:
      invalid        -- unusable: unknown concept, no year, no number
      wrong_value    -- right concept and year, wrong number
      wrong_year     -- the number belongs to the same concept in another year
      wrong_concept  -- the number belongs to another concept
      unlabelled     -- no label has this number, but the passage prints it:
                        a misread row, or a fact the labeller missed
      not_in_passage -- no label has it and the passage does not print it
    For an expected fact the model did not get: missed.
    """
    exp, _ = _keyed(expected)
    out = {k: 0 for k in (
        "invalid", "wrong_value", "wrong_year", "wrong_concept",
        "unlabelled", "not_in_passage", "missed", "sign_flipped",
    )}
    pred: dict[tuple[str, int], float] = {}
    for fact in predicted:
        norm = normalise_fact(fact)
        if norm is None:
            out["invalid"] += 1
            continue
        pred.setdefault((norm[0], norm[1]), norm[2])
    pred, _, _ = _drop_ignored(pred, ignore, tol, verifiable)

    for key, value in pred.items():
        if key in exp and values_match(value, exp[key], tol):
            if value * exp[key] < 0:
                out["sign_flipped"] += 1
            continue
        if key in exp:
            out["wrong_value"] += 1
            continue
        same_number = [k for k, v in exp.items() if values_match(value, v, tol)]
        if any(k[0] == key[0] for k in same_number):
            out["wrong_year"] += 1
        elif same_number:
            out["wrong_concept"] += 1
        elif value_in_passage(passage, value):
            out["unlabelled"] += 1
        else:
            out["not_in_passage"] += 1

    for key, value in exp.items():
        if not (key in pred and values_match(pred[key], value, tol)):
            out["missed"] += 1
    return out
