"""Why one arm loses to another, error item by error item.

`diagnose` says what kind each error is (missed, unlabelled, wrong_value, ...).
This module says why: every error item of the analysed arm gets one cause,
named in `CAUSES`, and every count in the report traces to the items list
written beside it.

The causes were assigned by reading every case with an error beside the
baseline's answer to the same case. A reading is not a rule, so each
assignment is checked by one (`_check`): a cause that says "parent-company
statement" needs "Parent Company Only" in the passage, "net sales as revenue"
needs the figure on a net-sales row, and so on. A misfiled item fails the run
rather than moving a count. Omissions are the one cause split by rule alone:
an omitted label is "prose" when the line printing it has no table cell.

Label artifacts are separated from model errors. Three kinds were found:

- the label is XBRL's exact value but the passage prints it rounded in prose
  ("$1.4 billion" for 1,442,000,000), more than the scorer's 0.5% away;
- the passage prints a restated figure for its oldest year (Nvidia after the
  10:1 split, J&J recast after the Kenvue separation) that the label builder
  never made a candidate, so a correct reading scores as a false positive;
- the passage prints no year for the labelled figure.

No model is called and nothing is re-scored: items come from the same
`parse_extraction` and the same arithmetic as `diagnose`, and the per-case
counts are asserted equal to it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any

from .scoring import (
    _LEFT,
    _RIGHT,
    _drop_ignored,
    _keyed,
    diagnose,
    find_value,
    parse_extraction,
    printed_forms,
    value_in_passage,
    values_match,
)

KINDS = ("missed", "unlabelled", "wrong_value", "wrong_year", "not_in_passage")

# code: (side, short name, what it means)
CAUSES: dict[str, tuple[str, str, str]] = {
    "nci": ("model", "noncontrolling interests",
            "a figure that includes noncontrolling interests, or is theirs, given as the company's"),
    "net_sales": ("model", "net sales as total revenue",
                  "Costco's net sales given as total revenue, which adds membership fees"),
    "parent_only": ("model", "parent-company statements",
                    "UnitedHealth's parent-only Schedule I read as the consolidated company"),
    "segment": ("model", "segment table",
                "a segment column or sum of segments given as the company figure, or segment columns read as years"),
    "wrong_cell": ("model", "wrong row or column",
                   "a neighbouring row or column: a component of cash, a fair-value level, pension income, the non-GAAP column"),
    "year_header": ("model", "year from the wrong header",
                    "a row given the year of the balance date above it (equity roll-forward) or of the block header below it"),
    "multicolumn": ("model", "multi-year row misread",
                    "a row printing several years mapped to the wrong ones: columns reversed, or the beginning-of-year balance"),
    "guessed_year": ("model", "year guessed",
                     "a figure the passage prints without a year, reported under a year the model chose"),
    "shares": ("model", "shares outstanding",
               "shares outstanding skipped where they sit inside a row label or an equity-statement balance, or the issued count taken"),
    "op_skipped": ("model", "\"Earnings from operations\" skipped",
                   "UnitedHealth's operating income row skipped while the rows around it were read"),
    "omitted_table": ("model", "table figure omitted", "a labelled table figure not reported, for no reason above"),
    "omitted_prose": ("model", "prose figure omitted", "a labelled figure in a sentence not reported"),
    "extra_digits": ("model", "digits added to a rounded figure",
                     "\"$6.9 billion\" given as 6,866,000,000, which misses a label the rounded figure would have matched"),
    "computed": ("model", "figure not printed",
                 "a figure the passage does not print: worked out from a growth rate or margin, or made up"),
    "label_rounding": ("label", "label beyond rounding",
                       "the passage prints the figure rounded in prose, more than 0.5% from the exact label"),
    "label_restated": ("label", "restated figure not a label",
                       "the passage prints a later restatement of its oldest year, which the label builder never considered"),
    "label_no_year": ("label", "year not printed",
                      "the passage prints the labelled figure with no year above it"),
}

# The causes in five groups, for the summary.
FAMILIES: dict[str, tuple[str, ...]] = {
    "definitions the prompt states": ("nci", "net_sales", "parent_only", "segment"),
    "which year": ("year_header", "multicolumn", "guessed_year"),
    "left out": ("shares", "op_skipped", "omitted_table", "omitted_prose"),
    "not printed": ("extra_digits", "computed"),
    "wrong cell": ("wrong_cell",),
    "label artifacts": ("label_rounding", "label_restated", "label_no_year"),
}

# The reading. Case-level cause, then item-level overrides keyed
# (case, kind, concept, year). Omissions are left to `_omitted`.
OMITTED = "omitted"
CASE_CAUSE: dict[str, str] = {
    # ADBE
    "0000796343-24-000006:157": "shares",
    "0000796343-24-000006:160": OMITTED,
    "0000796343-24-000006:185": "computed",
    "0000796343-24-000006:194": "wrong_cell",
    "0000796343-25-000004:163": "shares",
    "0000796343-25-000004:191": "segment",
    "0000796343-26-000003:177": "wrong_cell",
    "0000796343-26-000003:178": "guessed_year",
    # CMCSA
    "0001166691-24-000011:197": "nci",
    "0001166691-24-000011:202": "nci",
    "0001166691-24-000011:203": "nci",
    "0001166691-25-000011:181": "nci",
    "0001166691-25-000011:186": "nci",
    "0001166691-25-000011:187": "nci",
    "0001628280-26-004994:182": "nci",
    "0001628280-26-004994:187": "nci",
    "0001628280-26-004994:188": "nci",
    # COST
    "0000909832-24-000049:61": "net_sales",
    "0000909832-24-000049:62": "net_sales",
    "0000909832-24-000049:83": "net_sales",
    "0000909832-24-000049:86": OMITTED,
    "0000909832-24-000049:87": "year_header",
    "0000909832-24-000049:89": OMITTED,
    "0000909832-25-000101:134": "net_sales",
    "0000909832-25-000101:66": "net_sales",
    "0000909832-25-000101:86": "net_sales",
    "0000909832-25-000101:88": OMITTED,
    "0000909832-25-000101:91": OMITTED,
    "0000909832-26-000093:134": "net_sales",
    "0000909832-26-000093:64": "net_sales",
    "0000909832-26-000093:87": "shares",
    # JNJ
    "0000200406-24-000013:118": "computed",
    "0000200406-24-000013:160": "label_restated",
    "0000200406-24-000013:166": "multicolumn",
    "0000200406-24-000013:225": OMITTED,
    "0000200406-25-000038:116": "computed",
    "0000200406-26-000016:121": "computed",
    # LMT
    "0000936468-24-000010:134": "extra_digits",
    "0000936468-24-000010:148": "label_rounding",
    "0000936468-24-000010:151": OMITTED,
    "0000936468-24-000010:152": "multicolumn",
    "0000936468-24-000010:154": "label_rounding",
    "0000936468-24-000010:155": "label_rounding",
    "0000936468-24-000010:193": "multicolumn",
    "0000936468-24-000010:237": "guessed_year",
    "0000936468-24-000010:271": "shares",
    "0000936468-24-000010:272": "shares",
    "0000936468-25-000009:133": "extra_digits",
    "0000936468-25-000009:152": "multicolumn",
    "0000936468-25-000009:193": "multicolumn",
    "0000936468-25-000009:214": OMITTED,
    "0000936468-25-000009:238": "year_header",
    "0000936468-25-000009:239": "guessed_year",
    "0000936468-25-000009:271": "shares",
    "0001628280-26-004195:137": "extra_digits",
    "0001628280-26-004195:151": "label_rounding",
    "0001628280-26-004195:157": "label_rounding",
    "0001628280-26-004195:187": "multicolumn",
    "0001628280-26-004195:234": "year_header",
    "0001628280-26-004195:235": "guessed_year",
    "0001628280-26-004195:253": "wrong_cell",
    # NVDA
    "0001045810-24-000029:165": "multicolumn",
    "0001045810-24-000029:166": "year_header",
    "0001045810-24-000029:218": "segment",
    "0001045810-25-000023:169": "label_restated",
    "0001045810-25-000023:170": "label_restated",
    "0001045810-25-000023:173": "year_header",
    "0001045810-25-000023:197": "label_restated",
    "0001045810-25-000023:198": "wrong_cell",
    "0001045810-26-000021:169": "shares",
    "0001045810-26-000021:170": "shares",
    "0001045810-26-000021:219": "segment",
    # PG
    "0000080424-24-000083:136": "nci",
    "0000080424-24-000083:139": "year_header",
    "0000080424-24-000083:81": "nci",
    "0000080424-25-000076:138": "nci",
    "0000080424-25-000076:155": "nci",
    "0000080424-25-000076:156": "nci",
    "0000080424-25-000076:157": "nci",
    "0000080424-25-000076:90": "computed",
    "0000080424-26-000103:125": "wrong_cell",
    "0000080424-26-000103:138": "nci",
    "0000080424-26-000103:155": "nci",
    "0000080424-26-000103:156": "nci",
    "0000080424-26-000103:157": "nci",
    # SLB
    "0000950170-24-006884:107": "nci",
    "0000950170-25-007638:107": "nci",
    "0000950170-25-007638:111": "nci",
    "0001193125-26-021017:110": "nci",
    # UNH
    "0000731766-24-000081:107": "op_skipped",
    "0000731766-24-000081:115": "computed",
    "0000731766-24-000081:135": "shares",
    "0000731766-24-000081:165": "wrong_cell",
    "0000731766-24-000081:194": OMITTED,
    "0000731766-24-000081:195": "guessed_year",
    "0000731766-24-000081:219": "parent_only",
    "0000731766-24-000081:221": "parent_only",
    "0000731766-25-000063:108": OMITTED,
    "0000731766-25-000063:109": "op_skipped",
    "0000731766-25-000063:138": "shares",
    "0000731766-25-000063:144": OMITTED,
    "0000731766-25-000063:198": OMITTED,
    "0000731766-25-000063:199": "year_header",
    "0000731766-25-000063:223": "parent_only",
    "0000731766-26-000062:110": OMITTED,
    "0000731766-26-000062:112": "op_skipped",
    "0000731766-26-000062:142": "shares",
    "0000731766-26-000062:148": OMITTED,
    "0000731766-26-000062:212": "year_header",
    "0000731766-26-000062:213": "year_header",
    "0000731766-26-000062:231": "parent_only",
    "0000731766-26-000062:232": "parent_only",
    # V
    "0001403161-24-000058:137": "extra_digits",
    "0001403161-24-000058:230": "year_header",
    "0001403161-25-000089:243": "year_header",
}

NI, EPS, REV = "net income", "diluted earnings per share", "total revenue"
CASH, ASSETS, OPINC = "cash and cash equivalents", "total assets", "operating income"
EQUITY, SHARES = "total stockholders' equity", "common shares outstanding"

ITEM_CAUSE: dict[tuple[str, str, str, int], str] = {
    # ADBE: the 7,613 row has no year; gross profit invented, revenue skipped
    ("0000796343-26-000003:178", "missed", CASH, 2024): "label_no_year",
    **{("0000796343-24-000006:185", "missed", REV, y): OMITTED for y in (2021, 2022, 2023)},
    # COST: the net-sales errors, beside an omitted prose figure and a computed one
    ("0000909832-25-000101:66", "missed", NI, 2024): OMITTED,
    ("0000909832-25-000101:66", "missed", EPS, 2024): OMITTED,
    ("0000909832-26-000093:64", "not_in_passage", REV, 2025): "computed",
    # LMT: rounded prose, where a label was or was not reachable
    ("0000936468-24-000010:134", "wrong_value", NI, 2022): "label_rounding",
    ("0000936468-24-000010:134", "missed", NI, 2022): "label_rounding",
    ("0000936468-24-000010:134", "missed", EPS, 2023): OMITTED,
    ("0000936468-24-000010:134", "missed", EPS, 2022): OMITTED,
    ("0000936468-25-000009:133", "wrong_value", NI, 2024): "label_rounding",
    ("0000936468-25-000009:133", "missed", NI, 2024): "label_rounding",
    ("0001628280-26-004195:137", "wrong_value", NI, 2024): "label_rounding",
    ("0001628280-26-004195:137", "missed", NI, 2024): "label_rounding",
    ("0001628280-26-004195:137", "missed", EPS, 2025): OMITTED,
    ("0001628280-26-004195:137", "missed", EPS, 2024): OMITTED,
    ("0000936468-24-000010:193", "missed", NI, 2021): OMITTED,
    # LMT: sales tables, the first with no year, the next under a printed one
    ("0000936468-24-000010:237", "missed", REV, 2021): "label_no_year",
    ("0000936468-25-000009:239", "missed", REV, 2022): "label_no_year",
    ("0001628280-26-004195:235", "missed", REV, 2023): "label_no_year",
    ("0000936468-25-000009:238", "wrong_year", REV, 2024): "guessed_year",
    ("0000936468-25-000009:238", "missed", REV, 2023): "label_no_year",
    ("0001628280-26-004195:234", "wrong_year", REV, 2025): "guessed_year",
    ("0001628280-26-004195:234", "missed", REV, 2024): "label_no_year",
    # NVDA: share counts beside roll-forward year errors, an equity balance skipped
    **{("0001045810-24-000029:166", "missed", SHARES, y): "shares" for y in (2023, 2024)},
    **{("0001045810-25-000023:173", "missed", SHARES, y): "shares" for y in (2024, 2025)},
    ("0001045810-26-000021:170", "missed", EQUITY, 2026): OMITTED,
    # PG: Core EPS column taken; the as-reported rows beside it skipped
    ("0000080424-26-000103:125", "missed", OPINC, 2025): OMITTED,
    ("0000080424-26-000103:125", "missed", NI, 2025): OMITTED,
    ("0000080424-25-000076:155", "missed", REV, 2025): OMITTED,
    ("0000080424-26-000103:155", "missed", REV, 2026): OMITTED,
    # SLB: stockholders' equity skipped beside the NCI net income
    ("0001193125-26-021017:110", "missed", EQUITY, 2025): OMITTED,
    ("0001193125-26-021017:110", "missed", EQUITY, 2024): OMITTED,
    # UNH: stacked segment blocks
    ("0000731766-24-000081:195", "missed", ASSETS, 2022): "label_no_year",
    ("0000731766-24-000081:195", "missed", ASSETS, 2021): OMITTED,
    ("0000731766-25-000063:199", "missed", ASSETS, 2023): "label_no_year",
    ("0000731766-26-000062:213", "wrong_value", ASSETS, 2024): "segment",
    ("0000731766-26-000062:213", "missed", ASSETS, 2024): "segment",
    # V: the year block above the passage
    ("0001403161-24-000058:230", "missed", NI, 2023): "label_no_year",
    ("0001403161-25-000089:243", "missed", NI, 2024): "label_no_year",
}

# Defect B, which needs XBRL history to detect: the passages that print a
# restated oldest year the label builder never made a candidate.
RESTATED: set[tuple[str, str, int]] = {
    ("0001045810-25-000023:169", EPS, 2023),
    ("0001045810-25-000023:170", EPS, 2023),
    ("0001045810-25-000023:197", EPS, 2023),
    ("0000200406-24-000013:160", REV, 2021),
    ("0000200406-24-000013:160", "gross profit", 2021),
    ("0000200406-24-000013:160", "research and development expense", 2021),
    ("0001045810-25-000023:172", SHARES, 2023),
    ("0001045810-25-000023:173", SHARES, 2023),
}

_YEAR = re.compile(r"(?<![\d,.$])(?:19|20)\d{2}(?![\d,])")
_PROSE_FIGURE = re.compile(r"(\d+(?:\.\d+)?)\s*(billion|million)", re.I)


# ------------------------------------------------------------------ evidence
def locate(passage: str, value: float | None, concept: str) -> re.Match[str] | None:
    """Where the passage prints `value`: a table form first, then prose."""
    if value is None:
        return None
    for form in printed_forms(value, concept):
        match = re.search(_LEFT + re.escape(form) + _RIGHT, passage)
        if match:
            return match
    match = find_value(passage, value, concept)
    if match is None and concept != EPS:
        # `printed_forms` skips figures under 100 at their scale; an NCI row
        # printing "25" (millions) is still where the figure came from.
        for scale in (1_000_000, 1_000):
            scaled = abs(value) / scale
            if 1 <= scaled < 100 and scaled == round(scaled):
                match = re.search(_LEFT + re.escape(f"{round(scaled)}") + _RIGHT, passage)
                if match:
                    break
    return match


def scale_slip(passage: str, value: float, concept: str) -> int | float | None:
    """The factor, when a figure the passage does not print is a printed one
    times 10 or 100 (or divided by): "240" (millions) given as 2,400,000,000."""
    if printed_exactly(passage, value, concept) or value_in_passage(passage, value):
        return None
    for factor in (10, 100, 0.1, 0.01):
        if printed_exactly(passage, value / factor, concept):
            return factor
    return None


def line_of(passage: str, match: re.Match[str] | None, width: int = 220) -> str | None:
    """The table row holding a match, or the stretch of sentence around it."""
    if match is None:
        return None
    start = passage.rfind("\n", 0, match.start()) + 1
    end = passage.find("\n", match.end())
    end = len(passage) if end < 0 else end
    if end - start > width:
        start = max(start, match.start() - width // 2)
        end = min(end, match.end() + width // 2)
    return passage[start:end].strip()


def is_prose(passage: str, match: re.Match[str] | None) -> bool | None:
    """A figure is in prose when the line printing it has no table cell."""
    line = line_of(passage, match, width=10_000)
    return None if line is None else "|" not in line


def year_before(passage: str, match: re.Match[str] | None) -> bool | None:
    """Whether any year is printed before the start of the figure's row."""
    if match is None:
        return None
    row_start = passage.rfind("\n", 0, match.start()) + 1
    return bool(_YEAR.search(passage, 0, row_start))


def printed_exactly(passage: str, value: float, concept: str) -> bool:
    """Whether a table form of the value is printed. A rounded prose figure
    does not count: "$6.9 billion" does not print 6,866,000,000."""
    return any(re.search(_LEFT + re.escape(f) + _RIGHT, passage) for f in printed_forms(value, concept))


def rounding_unreachable(passage: str, value: float, concept: str) -> str | None:
    """The prose figure, when the passage prints this label only rounded and
    the rounded figure misses it by more than the scorer's tolerance."""
    if printed_exactly(passage, value, concept):
        return None
    match = find_value(passage, value, concept)
    if match is None:
        return None
    number, word = _PROSE_FIGURE.search(match.group(0)).groups()
    printed = float(number) * (1e9 if word.lower() == "billion" else 1e6)
    return None if values_match(printed, value) else match.group(0)


def rounds_to_prose(passage: str, value: float) -> str | None:
    """A prose figure ("$6.9 billion") the value rounds to, if one is printed."""
    for match in _PROSE_FIGURE.finditer(passage):
        number, word = match.groups()
        scale = 1e9 if word.lower() == "billion" else 1e6
        decimals = len(number.split(".")[1]) if "." in number else 0
        if round(abs(value) / scale, decimals) == float(number):
            return match.group(0)
    return None


# --------------------------------------------------------------------- items
def _scored_predictions(text: str, row: dict[str, Any]) -> dict[tuple[str, int], float]:
    pred, _ = _keyed(parse_extraction(text))
    pred, _, _ = _drop_ignored(pred, row["dropped"], 0.005, row.get("verifiable"))
    return pred


def error_items(text: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    """Each error `diagnose` counts for this answer, as one item."""
    exp, _ = _keyed(row["facts"])
    pred = _scored_predictions(text, row)
    items = []
    for key, value in pred.items():
        if key in exp and values_match(value, exp[key]):
            continue
        if key in exp:
            kind, label = "wrong_value", exp[key]
        else:
            same = [k for k, v in exp.items() if values_match(value, v)]
            if any(k[0] == key[0] for k in same):
                kind, label = "wrong_year", None
            elif same:
                kind, label = "wrong_concept", None
            elif value_in_passage(row["passage"], value):
                kind, label = "unlabelled", None
            else:
                kind, label = "not_in_passage", None
        items.append({"kind": kind, "concept": key[0], "year": key[1], "predicted": value, "label": label})
    for key, value in exp.items():
        if not (key in pred and values_match(pred[key], value)):
            items.append({"kind": "missed", "concept": key[0], "year": key[1],
                          "predicted": pred.get(key), "label": value})
    return items


def baseline_right(item: dict[str, Any], base: dict[tuple[str, int], float], exp: dict[tuple[str, int], float]) -> bool:
    """Whether the baseline arm avoids this error on the same case.

    For a label the arm missed or got wrong: the baseline reports it. For a
    figure the arm reported and no label has: the baseline does not report
    that (concept, year), or reports it right.
    """
    key = (item["concept"], item["year"])
    if item["kind"] in ("missed", "wrong_value"):
        return key in base and values_match(base[key], exp[key])
    return key not in base or (key in exp and values_match(base[key], exp[key]))


def cause_of(case_id: str, item: dict[str, Any], prose: bool | None) -> str:
    key = (case_id, item["kind"], item["concept"], item["year"])
    cause = ITEM_CAUSE.get(key, CASE_CAUSE.get(case_id))
    if cause is None:
        raise KeyError(f"no cause for {key}")
    if cause == OMITTED:
        return "omitted_prose" if prose else "omitted_table"
    return cause


# -------------------------------------------------------------------- checks
def _check(cause: str, item: dict[str, Any], row: dict[str, Any], labels_elsewhere: set[float]) -> bool:
    """The rule an assignment must satisfy. False fails the run."""
    p = row["passage"]
    pred, label = item["predicted"], item["label"]
    concept, kind = item["concept"], item["kind"]
    pred_printed = pred is not None and value_in_passage(p, pred) or (
        pred is not None and locate(p, pred, concept) is not None
    )
    pred_line = line_of(p, locate(p, pred, concept)) or ""
    label_match = locate(p, label, concept)
    if cause == "nci":
        # The passage names the interests, or the company's own figure for
        # this concept and year (here or in another passage) is different.
        own = labels_elsewhere | ({label} if label is not None else set())
        return concept in (NI, EQUITY) and pred is not None and (
            bool(re.search(r"noncontrolling|non-controlling", p, re.I))
            or (bool(own) and not any(values_match(pred, v) for v in own))
        )
    if cause == "net_sales":
        return concept == REV and row["ticker"] == "COST" and (
            pred is None or bool(re.search(r"net sales", pred_line, re.I)) or kind == "not_in_passage"
        )
    if cause == "parent_only":
        return "Parent Company Only" in p
    if cause == "segment":
        return bool(re.search(r"segment", p, re.I))
    if cause == "wrong_cell":
        return pred is not None and pred_printed
    if cause == "multicolumn":
        # The wrong figure is printed, or the skipped label sits on a row
        # printing several figures.
        multi = (line_of(p, label_match, width=10_000) or "").count("|") >= 2
        return (pred is not None and pred_printed) or (kind == "missed" and pred is None and multi)
    if cause == "year_header":
        return bool(_YEAR.search(p)) and (pred is None or pred_printed)
    if cause == "guessed_year":
        return kind == "wrong_year" and year_before(p, locate(p, pred, concept)) is False
    if cause == "shares":
        return concept == SHARES
    if cause == "op_skipped":
        return concept == OPINC and kind == "missed" and pred is None and bool(
            re.search(r"earnings from operations", line_of(p, label_match) or "", re.I)
        )
    if cause in ("omitted_table", "omitted_prose"):
        return kind == "missed" and pred is None and label_match is not None
    if cause == "extra_digits":
        return (pred is not None and not printed_exactly(p, pred, concept) and rounds_to_prose(p, pred) is not None
                and label is not None and rounding_unreachable(p, label, concept) is None)
    if cause == "computed":
        return pred is not None and not printed_exactly(p, pred, concept) and not value_in_passage(p, pred)
    if cause == "label_rounding":
        return label is not None and rounding_unreachable(p, label, concept) is not None
    if cause == "label_restated":
        return ((row["id"], concept, item["year"]) in RESTATED and item["year"] == row["filing_fy"] - 2
                and locate(p, pred, concept) is not None)
    if cause == "label_no_year":
        return kind == "missed" and year_before(p, label_match) is False
    raise KeyError(cause)


# -------------------------------------------------------------------- report
def _load_answers(path: str) -> dict[str, dict[str, Any]]:
    last: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                last[record["id"]] = record
    return last


def _sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _micro_f1(tp: int, n_pred: int, n_exp: int) -> float:
    return round(2 * tp / (n_pred + n_exp), 4) if n_pred + n_exp else 0.0


def _micro(rows: dict[str, dict[str, Any]], answers: dict[str, dict[str, Any]],
           set_aside: set[tuple[str, str, int]] = frozenset()) -> dict[str, Any]:
    """Pooled tp / n_pred / n_expected / F1, with some (case, concept, year)
    keys taken out of both the labels and the predictions."""
    tp = n_pred = n_exp = 0
    for case_id, row in rows.items():
        exp, _ = _keyed(row["facts"])
        pred = _scored_predictions(answers[case_id]["text"], row)
        exp = {k: v for k, v in exp.items() if (case_id, *k) not in set_aside}
        pred = {k: v for k, v in pred.items() if (case_id, *k) not in set_aside}
        tp += sum(1 for k, v in pred.items() if k in exp and values_match(v, exp[k]))
        n_pred += len(pred)
        n_exp += len(exp)
    return {"tp": tp, "n_pred": n_pred, "n_expected": n_exp, "f1": _micro_f1(tp, n_pred, n_exp)}


def _f1_if_fixed(items: list[dict[str, Any]], totals: dict[str, int]) -> float:
    """The arm's micro F1 with these items right: a missed label becomes a
    hit (a prediction is added where there was none), and a figure no label
    has is withdrawn."""
    gained = sum(1 for i in items if i["kind"] == "missed")
    added = sum(1 for i in items if i["kind"] == "missed" and i["predicted"] is None)
    withdrawn = sum(1 for i in items if i["kind"] in ("unlabelled", "wrong_year", "not_in_passage"))
    return _micro_f1(totals["tp"] + gained, totals["n_pred"] + added - withdrawn, totals["n_expected"])


def _summary(items: list[dict[str, Any]], totals: dict[str, int]) -> dict[str, Any]:
    kinds = Counter(i["kind"] for i in items)
    return {"items": len(items), **{k: kinds.get(k, 0) for k in KINDS},
            "baseline_right": sum(i["baseline_right"] for i in items),
            "cases": len({i["id"] for i in items}),
            "f1_if_fixed": _f1_if_fixed(items, totals)}


def rounded_prose_labels(path: str) -> dict[str, int]:
    """Labels in a dataset file whose passage prints them only as a rounded
    prose figure, and how many of those miss the label by more than 0.5%.
    The training target is the exact value either way."""
    prose_only = beyond = 0
    with open(path, encoding="utf-8") as fh:
        for row in map(json.loads, fh):
            for fact in row["facts"]:
                value, concept = float(fact["value"]), fact["concept"]
                if printed_exactly(row["passage"], value, concept):
                    continue
                if find_value(row["passage"], value, concept) is None:
                    continue
                prose_only += 1
                beyond += rounding_unreachable(row["passage"], value, concept) is not None
    return {"prose_only": prose_only, "beyond_tolerance": beyond}


def analyse(held_out: str, arm_path: str, base_path: str, train: str | None = None) -> dict[str, Any]:
    """Every error item of the arm at `arm_path`, with its cause and the
    baseline's result on it, plus the totals the doc quotes."""
    with open(held_out, encoding="utf-8") as fh:
        rows = {r["id"]: r for r in map(json.loads, fh)}
    arm, base = _load_answers(arm_path), _load_answers(base_path)

    labels_by_company: dict[tuple[str, str, int], set[float]] = defaultdict(set)
    for r in rows.values():
        for f in r["facts"]:
            labels_by_company[(r["ticker"], f["concept"], int(f["fiscal_year"]))].add(float(f["value"]))

    items: list[dict[str, Any]] = []
    failures: list[str] = []
    totals = {"tp": 0, "n_pred": 0, "n_expected": 0}
    diagnosis: Counter[str] = Counter()
    for case_id, row in rows.items():
        text = arm[case_id]["text"]
        case_items = error_items(text, row)
        expected = diagnose(parse_extraction(text), row["facts"], row["passage"],
                            ignore=row["dropped"], verifiable=row.get("verifiable"))
        got = Counter(i["kind"] for i in case_items)
        assert all(got[k] == expected[k] for k in (*KINDS, "wrong_concept")), (case_id, got, expected)
        diagnosis.update({k: expected[k] for k in KINDS})

        exp, _ = _keyed(row["facts"])
        pred = _scored_predictions(text, row)
        tp = sum(1 for k, v in pred.items() if k in exp and values_match(v, exp[k]))
        totals["tp"] += tp
        totals["n_pred"] += len(pred)
        totals["n_expected"] += len(exp)

        base_pred = _scored_predictions(base[case_id]["text"], row)
        right_by_concept = Counter(k[0] for k, v in pred.items() if k in exp and values_match(v, exp[k]))
        for item in case_items:
            p = row["passage"]
            focus = item["label"] if item["kind"] == "missed" else item["predicted"]
            focus_match = locate(p, focus, item["concept"])
            prose = is_prose(p, focus_match)
            cause = cause_of(case_id, item, prose)
            elsewhere = labels_by_company[(row["ticker"], item["concept"], item["year"])] - {
                float(f["value"]) for f in row["facts"]
            }
            if not _check(cause, item, row, elsewhere):
                failures.append(f"{case_id} {item['kind']} {item['concept']}/{item['year']} -> {cause}")
            items.append({
                "id": case_id,
                "ticker": row["ticker"],
                "negative": not row["facts"],
                "kind": item["kind"],
                "concept": item["concept"],
                "year": item["year"],
                "predicted": item["predicted"],
                "label": item["label"],
                "baseline": base_pred.get((item["concept"], item["year"])),
                "baseline_right": baseline_right(item, base_pred, exp),
                "cause": cause,
                "side": CAUSES[cause][0],
                "prose": prose,
                "oldest_year": item["year"] == row["filing_fy"] - 2,
                # The arm got this concept right for another year of the case:
                # with `oldest_year`, a row read but for its last column.
                "concept_right_other_years": right_by_concept[item["concept"]],
                "scale_slip": None if item["predicted"] is None else scale_slip(p, item["predicted"], item["concept"]),
                "line": line_of(p, focus_match),
                "label_line": line_of(p, locate(p, item["label"], item["concept"])),
            })

    if failures:
        raise AssertionError("assignments that break their cause's rule:\n" + "\n".join(failures))

    by_cause: dict[str, dict[str, Any]] = {}
    for code, (side, name, meaning) in CAUSES.items():
        mine = [i for i in items if i["cause"] == code]
        if mine:
            by_cause[code] = {"side": side, "name": name, "meaning": meaning, **_summary(mine, totals),
                              "companies": dict(Counter(i["ticker"] for i in mine).most_common())}
    by_family = {
        family: {"causes": list(codes), **_summary([i for i in items if i["cause"] in codes], totals)}
        for family, codes in FAMILIES.items()
    }
    by_side = {side: _summary([i for i in items if i["side"] == side], totals) for side in ("model", "label")}
    by_company = {
        t: _summary([i for i in items if i["ticker"] == t], totals)
        for t, _ in Counter(i["ticker"] for i in items).most_common()
    }
    by_place = {
        place: _summary([i for i in items if i["prose"] is flag], totals)
        for place, flag in (("prose", True), ("table", False), ("not printed", None))
    }

    # Both arms with the label artifacts taken out of the key: the labels and
    # any prediction for the same (case, concept, year), for both alike.
    artifact_keys = {(i["id"], i["concept"], i["year"]) for i in items if i["side"] == "label"} | RESTATED
    set_aside = {
        "keys": len(artifact_keys),
        "arm": _micro(rows, arm, artifact_keys),
        "baseline": _micro(rows, base, artifact_keys),
    }
    set_aside["gap"] = round(set_aside["baseline"]["f1"] - set_aside["arm"]["f1"], 4)
    baseline_micro = _micro(rows, base)

    # The baseline's own errors, against the same artifacts.
    artifact_cause = {(i["id"], i["concept"], i["year"]): i["cause"] for i in items if i["side"] == "label"}
    artifact_cause.update({key: "label_restated" for key in RESTATED})
    base_kinds: Counter[str] = Counter()
    base_on_artifacts: dict[str, Counter[str]] = defaultdict(Counter)
    base_not_printed = 0
    for case_id, row in rows.items():
        for item in error_items(base[case_id]["text"], row):
            base_kinds[item["kind"]] += 1
            cause = artifact_cause.get((case_id, item["concept"], item["year"]))
            if cause:
                base_on_artifacts[cause][item["kind"]] += 1
            elif item["kind"] != "missed" and not printed_exactly(row["passage"], item["predicted"], item["concept"]) \
                    and not value_in_passage(row["passage"], item["predicted"]):
                base_not_printed += 1
    baseline_errors = {
        "diagnosis": {k: base_kinds[k] for k in KINDS},
        "on_label_artifacts": {c: dict(k) for c, k in base_on_artifacts.items()},
        "not_printed_outside_artifacts": base_not_printed,
    }

    counted = Counter(i["kind"] for i in items)
    assert all(counted[k] == diagnosis[k] for k in KINDS), (counted, diagnosis)
    return {
        "inputs": {
            "held_out": held_out, "held_out_sha256": _sha256(held_out),
            "arm": arm_path, "arm_sha256": _sha256(arm_path),
            "baseline": base_path, "baseline_sha256": _sha256(base_path),
            **({"train": train, "train_sha256": _sha256(train)} if train else {}),
        },
        "arm_micro": {**totals, "f1": _micro_f1(totals["tp"], totals["n_pred"], totals["n_expected"])},
        "baseline_micro": baseline_micro,
        "baseline_errors": baseline_errors,
        "gap": round(baseline_micro["f1"] - _micro_f1(totals["tp"], totals["n_pred"], totals["n_expected"]), 4),
        "diagnosis": {k: diagnosis[k] for k in KINDS},
        "negatives_with_facts": sorted({i["id"] for i in items if i["negative"]}),
        "f1_if_baseline_right_items_fixed": _f1_if_fixed([i for i in items if i["baseline_right"]], totals),
        "labels_set_aside": set_aside,
        "by_side": by_side,
        "by_family": by_family,
        "by_company": by_company,
        "by_place": by_place,
        "by_cause": by_cause,
        "omitted_table_last_column": sum(
            1 for i in items if i["cause"] == "omitted_table" and i["oldest_year"] and i["concept_right_other_years"]
        ),
        "scale_slips": [i for i in items if i["scale_slip"]],
        "rounded_prose_labels": {
            "held_out": rounded_prose_labels(held_out),
            **({"train": rounded_prose_labels(train)} if train else {}),
        },
        "items": items,
    }
