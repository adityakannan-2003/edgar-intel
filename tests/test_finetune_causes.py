"""The error-cause helpers: what counts as printed, prose, rounded, a scale slip."""

from __future__ import annotations

from edgar_intel.finetune.causes import (
    error_items,
    is_prose,
    locate,
    printed_exactly,
    rounding_unreachable,
    rounds_to_prose,
    scale_slip,
    year_before,
)
from edgar_intel.finetune.scoring import diagnose, parse_extraction

CASH, NI = "cash and cash equivalents", "net income"


def test_rounded_prose_label_beyond_tolerance():
    passage = "As of December 31, 2023, we had cash and cash equivalents of $1.4 billion."
    # 1.4 billion is 2.9% from 1,442,000,000: no reading of this passage matches.
    assert rounding_unreachable(passage, 1_442_000_000, CASH) == "1.4 billion"
    # 6.9 billion is 0.3% from 6,920,000,000: the rounded figure matches.
    assert rounding_unreachable("net earnings of $6.9 billion", 6_920_000_000, NI) is None
    # A table form makes the label reachable whatever the prose says.
    assert rounding_unreachable("$1.4 billion\nCash | 1,442", 1_442_000_000, CASH) is None


def test_added_digits_are_not_printed():
    passage = "We reported net earnings of $6.9 billion ($27.55 per share) in 2023."
    assert not printed_exactly(passage, 6_866_000_000, NI)
    assert rounds_to_prose(passage, 6_866_000_000) == "6.9 billion"
    assert rounds_to_prose(passage, 6_700_000_000) is None


def test_prose_and_table_and_year_before():
    passage = "Sales by contract type\nTotal sales | $ | 27,474 | $ | 67,571\n70 2023\nTotal sales | $ | 65,984"
    first = locate(passage, 67_571_000_000, "total revenue")
    later = locate(passage, 65_984_000_000, "total revenue")
    assert is_prose(passage, first) is False
    assert year_before(passage, first) is False
    assert year_before(passage, later) is True
    assert is_prose("Net income rose to $5.0 billion.", locate("Net income rose to $5.0 billion.", 5e9, NI))


def test_small_figures_are_located_on_their_row():
    passage = "Noncontrolling interests\nNet income (loss) | (15) | 9 | 25"
    assert locate(passage, 25_000_000, NI) is not None


def test_scale_slip():
    passage = "Gross profit | $ | 12,281 | $ | 2,920 | $ | 240 | $ | 15,441"
    assert scale_slip(passage, 2_400_000_000, "gross profit") == 10
    assert scale_slip(passage, 240_000_000, "gross profit") is None


def test_error_items_agree_with_diagnose():
    row = {
        "passage": "Cash and cash equivalents at end of year | $ | 2,483 | $ | 1,442 | $ | 2,547\nNet earnings | 9,999",
        "facts": [
            {"concept": CASH, "fiscal_year": 2024, "value": 2_483_000_000},
            {"concept": CASH, "fiscal_year": 2023, "value": 1_442_000_000},
            {"concept": CASH, "fiscal_year": 2022, "value": 2_547_000_000},
        ],
        "dropped": [],
        "verifiable": [[CASH, y] for y in (2022, 2023, 2024)] + [[NI, 2024]],
    }
    text = ('[{"concept":"cash and cash equivalents","fiscal_year":2022,"value":2483000000},'
            '{"concept":"cash and cash equivalents","fiscal_year":2024,"value":2547000000},'
            '{"concept":"net income","fiscal_year":2024,"value":9999000000}]')
    items = error_items(text, row)
    kinds = {k: sum(1 for i in items if i["kind"] == k) for k in ("missed", "wrong_value", "unlabelled")}
    expected = diagnose(parse_extraction(text), row["facts"], row["passage"], verifiable=row["verifiable"])
    assert kinds == {k: expected[k] for k in kinds} == {"missed": 3, "wrong_value": 2, "unlabelled": 1}
