"""The fine-tune labeller, scorer and split: the parts a benchmark number rests on."""

from __future__ import annotations

import pytest

from edgar_intel.finetune.dataset import (
    FT_SECTORS,
    INSTRUCTION,
    Example,
    candidate_facts,
    label_passage,
    split,
)
from edgar_intel.finetune.scoring import (
    CONCEPTS,
    diagnose,
    find_value,
    parse_extraction,
    parse_ok,
    printed_forms,
    score_extraction,
)


def fact(tag: str, year: int, value: float, prior: float | None = None) -> dict:
    return {"tag": tag, "fiscal_year": year, "value": value, "prior_year_value": prior}


class TestPrintedForms:
    def test_eps_is_matched_to_the_cent_and_only_there(self):
        # The first labeller searched EPS 6.13 as "6": 18,480 of 19,199 labels.
        assert printed_forms(6.13, "diluted earnings per share") == ["6.13"]

    def test_only_scales_the_value_is_an_exact_multiple_of(self):
        forms = printed_forms(383_285_000_000, "total revenue")
        assert forms == ["383,285,000,000", "383,285,000", "383,285"]
        # 606.2 million shares is never printed "606" in a millions table.
        assert "606" not in printed_forms(606_200_000, "common shares outstanding")

    def test_small_cells_are_never_bare_matches(self):
        assert printed_forms(42_000_000, "total revenue") == ["42,000,000", "42,000"]


class TestFindValue:
    @pytest.mark.parametrize(
        "passage",
        [
            "Total net sales | 383,285 | 394,328",
            "Total net sales | $383,285",
            "net sales were $383.3 billion in 2023",
        ],
    )
    def test_finds_the_printed_figure(self, passage):
        assert find_value(passage, 383_285_000_000, "total revenue")

    @pytest.mark.parametrize(
        "passage",
        ["grew 6.13% year over year", "Diluted | 16.13", "Diluted | 6.135"],
    )
    def test_eps_needs_a_whole_number_boundary(self, passage):
        assert find_value(passage, 6.13, "diluted earnings per share") is None

    def test_no_match_inside_a_longer_number(self):
        assert find_value("Other | 1,383,285", 383_285_000_000, "total revenue") is None


class TestLabelling:
    def test_window_is_the_years_the_filing_prints(self):
        facts = [fact("NetIncomeLoss", y, 1000 + y) for y in range(2009, 2025)]
        cands = candidate_facts(facts, 2024)
        assert sorted(y for _c, y in cands) == [2022, 2023, 2024]

    def test_prior_year_value_is_a_second_way_to_print_a_year(self):
        facts = [
            fact("NetIncomeLoss", 2023, 100_000_000),
            fact("NetIncomeLoss", 2024, 120_000_000, prior=101_000_000),
        ]
        cands = candidate_facts(facts, 2024)
        assert cands[("net income", 2023)] == [100_000_000, 101_000_000]

    def test_two_revenue_tags_collapse_to_the_preferred_one(self):
        facts = [
            fact("Revenues", 2024, 500_000_000),
            fact("RevenueFromContractWithCustomerExcludingAssessedTax", 2024, 480_000_000),
        ]
        assert candidate_facts(facts, 2024)[("total revenue", 2024)] == [480_000_000]

    def test_a_figure_on_a_line_that_names_its_concept_is_a_label(self):
        cands = {("total assets", 2024): [352_583_000_000.0]}
        labels, dropped, _ = label_passage("Total assets | $ | 352,583", cands)
        assert labels == [{"concept": "total assets", "fiscal_year": 2024, "value": 352583000000}]
        assert dropped == []

    def test_the_accounting_identity_is_not_a_statement_of_total_assets(self):
        cands = {("total assets", 2024): [352_583_000_000.0]}
        passage = "Total equity | 62,146\nTotal liabilities and equity | $ | 352,583"
        labels, dropped, _ = label_passage(passage, cands)
        assert labels == [] and len(dropped) == 1

    def test_a_heading_within_three_lines_anchors_a_row(self):
        cands = {("diluted earnings per share", 2024): [15.51]}
        passage = "Earnings per share attributable to common shareholders:\nBasic | 15.64\nDiluted | $ | 15.51"
        labels, _, _ = label_passage(passage, cands)
        assert labels and labels[0]["value"] == 15.51

    def test_prose_needs_the_concept_in_the_same_sentence(self):
        # From the first audit: LMT's capitalized software, labelled as cash.
        cands = {("cash and cash equivalents", 2023): [1_442_000_000.0]}
        passage = (
            "Our cash position is discussed below. Costs are amortized over two to 15 "
            "years. As of December 31, 2023, capitalized software totaled $1.4 billion."
        )
        labels, dropped, _ = label_passage(passage, cands)
        assert labels == [] and len(dropped) == 1
        labels, _, _ = label_passage("At year end we had cash and cash equivalents of $1.4 billion.", cands)
        assert len(labels) == 1

    def test_a_concept_word_after_a_prose_figure_counts(self):
        cands = {("diluted earnings per share", 2024): [16.56]}
        passage = "Net income for 2024 was " + "x" * 300 + " $7,367, or $16.56 per diluted share."
        labels, _, _ = label_passage(passage, cands)
        assert len(labels) == 1

    def test_the_section_title_is_not_an_anchor(self):
        # From the first audit: a buyback authorisation labelled as equity.
        cands = {("total stockholders' equity", 2022): [9_266_000_000.0]}
        passage = "[5 Market for Registrant's Common Equity]The remaining authority under our share repurchase program was $9.3 billion."
        labels, dropped, _ = label_passage(passage, cands)
        assert labels == [] and len(dropped) == 1

    def test_two_facts_on_one_printed_number_make_the_passage_ambiguous(self):
        cands = {
            ("common shares outstanding", 2024): [2_464_000_000.0],
            ("common shares outstanding", 2023): [2_464_000_000.0],
        }
        _, _, ambiguous = label_passage("shares outstanding | 2,464", cands)
        assert ambiguous

    def test_labels_follow_reading_order(self):
        cands = {
            ("total revenue", 2023): [383_285_000_000.0],
            ("total revenue", 2024): [391_035_000_000.0],
        }
        labels, _, _ = label_passage("Total net sales | 391,035 | 383,285", cands)
        assert [f["fiscal_year"] for f in labels] == [2024, 2023]


class TestExampleFormat:
    def test_every_concept_is_in_the_instruction(self):
        assert all(c in INSTRUCTION for c in CONCEPTS)
        assert len(CONCEPTS) == 11

    def test_the_fiscal_year_rule_is_stated(self):
        # Home Depot calls the year ending January 2024 "fiscal 2023"; XBRL and
        # the labels call it 2024. A model can only follow a rule it is given.
        assert "first week of January" in INSTRUCTION

    def test_header_carries_the_filing_year_and_window(self):
        ex = Example(
            id="a:1", ticker="AAPL", company="Apple Inc.", accession="a",
            filing_fy=2024, period_end="2024-09-28", passage="Total net sales | 391,035",
            facts=[{"concept": "total revenue", "fiscal_year": 2024, "value": 391035000000}],
        )
        user = ex.to_chat()["messages"][1]["content"]
        assert "fiscal year 2024" in user and "2022, 2023, 2024" in user
        assert user.endswith("Total net sales | 391,035")
        assert parse_extraction(ex.answer()) == ex.facts


class TestScoring:
    expected = [{"concept": "total revenue", "fiscal_year": 2024, "value": 391_035_000_000}]

    def test_a_raw_xbrl_tag_scores_as_its_concept(self):
        pred = [{"tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "fiscal_year": 2024, "value": 391_035_000_000}]
        assert score_extraction(pred, self.expected)["f1"] == 1.0

    def test_a_value_as_printed_matches_at_its_scale(self):
        pred = [{"concept": "Total Revenue", "fiscal_year": "FY2024", "value": "391,035"}]
        assert score_extraction(pred, self.expected)["f1"] == 1.0

    def test_sign_is_not_scored_but_is_diagnosed(self):
        exp = [{"concept": "net income", "fiscal_year": 2024, "value": -18_756_000_000}]
        pred = [{"concept": "net income", "fiscal_year": 2024, "value": 18_756}]
        assert score_extraction(pred, exp)["f1"] == 1.0
        assert diagnose(pred, exp, "Net loss | (18,756)")["sign_flipped"] == 1

    def test_an_unusable_fact_is_a_false_positive(self):
        pred = self.expected + [{"concept": "ebitda", "fiscal_year": 2024, "value": 1}]
        score = score_extraction(pred, self.expected)
        assert score["precision"] == 0.5 and score["invalid"] == 1

    def test_an_ignored_fact_is_neither_credited_nor_penalised(self):
        ignore = [{"concept": "total assets", "fiscal_year": 2024, "value": 352_583_000_000}]
        pred = self.expected + [{"concept": "total assets", "fiscal_year": 2024, "value": 352_583}]
        score = score_extraction(pred, self.expected, ignore=ignore)
        assert score["precision"] == 1.0 and score["ignored"] == 1

    def test_counts_for_micro_f1(self):
        score = score_extraction([], self.expected)
        assert (score["tp"], score["n_pred"], score["n_expected"]) == (0, 0, 1)


class TestParsing:
    @pytest.mark.parametrize(
        "text",
        [
            '[{"concept":"net income","fiscal_year":2024,"value":1}]',
            '```json\n[{"concept":"net income","fiscal_year":2024,"value":1}]\n```',
            '<think>maybe [x]?</think>[{"concept":"net income","fiscal_year":2024,"value":1}]',
            'See note [1].\n[{"concept":"net income","fiscal_year":2024,"value":1}]',
            '{"facts": [{"concept":"net income","fiscal_year":2024,"value":1}]}',
        ],
    )
    def test_finds_the_array(self, text):
        assert parse_extraction(text)[0]["concept"] == "net income"

    def test_an_empty_answer_parses_and_garbage_does_not(self):
        assert parse_extraction("[]") == [] and parse_ok("[]")
        assert parse_extraction("I could not find any facts.") == []
        assert not parse_ok("I could not find any facts.")


class TestDiagnosis:
    exp = [
        {"concept": "total revenue", "fiscal_year": 2024, "value": 391_035_000_000},
        {"concept": "total revenue", "fiscal_year": 2023, "value": 383_285_000_000},
    ]
    passage = "Total net sales | 391,035 | 383,285\nCost of sales | 210,352"

    def test_each_kind_of_error_is_named(self):
        pred = [
            {"concept": "total revenue", "fiscal_year": 2022, "value": 391_035},  # wrong_year
            {"concept": "gross profit", "fiscal_year": 2023, "value": 383_285},   # wrong_concept
            {"concept": "net income", "fiscal_year": 2024, "value": 210_352},     # unlabelled
            {"concept": "net income", "fiscal_year": 2023, "value": 999_999},     # not_in_passage
        ]
        d = diagnose(pred, self.exp, self.passage)
        assert (d["wrong_year"], d["wrong_concept"], d["unlabelled"], d["not_in_passage"]) == (1, 1, 1, 1)
        assert d["missed"] == 2


class TestSplit:
    def _examples(self) -> list[Example]:
        return [
            Example(id=f"{t}:{i}", ticker=t, company=t, accession=t, filing_fy=2024,
                    period_end="2024-12-31", passage="p", facts=[], sector=s)
            for s, tickers in FT_SECTORS.items() for t in tickers for i in range(3)
        ]

    def test_company_disjoint_with_one_company_held_out_per_sector(self):
        train, held = split(self._examples())
        train_t, held_t = {e.ticker for e in train}, {e.ticker for e in held}
        assert not train_t & held_t
        assert len(held_t) == len(FT_SECTORS)
        for tickers in FT_SECTORS.values():
            assert len(held_t & set(tickers)) == 1

    def test_split_is_deterministic(self):
        a = {e.ticker for e in split(self._examples())[1]}
        b = {e.ticker for e in split(self._examples())[1]}
        assert a == b


class TestVerifiable:
    def test_an_uncheckable_prediction_is_set_aside_not_scored(self):
        # JNJ tags no StockholdersEquity, so its printed equity cannot be checked.
        expected = [{"concept": "total revenue", "fiscal_year": 2024, "value": 88_821_000_000}]
        pred = expected + [{"concept": "total stockholders' equity", "fiscal_year": 2024, "value": 71_490}]
        verifiable = [["total revenue", 2024], ["total revenue", 2023]]
        score = score_extraction(pred, expected, verifiable=verifiable)
        assert score["precision"] == 1.0 and score["unverifiable"] == 1
        assert score_extraction(pred, expected)["precision"] == 0.5

    def test_a_whole_number_does_not_count_as_printing_a_decimal(self):
        from edgar_intel.finetune.scoring import value_in_passage

        assert not value_in_passage("[8 Financial Statements]", 8.28)
        assert value_in_passage("Diluted | $ 8.28", 8.28)

    def test_the_instruction_gives_the_xbrl_definitions(self):
        assert "noncontrolling" in INSTRUCTION and "membership fees" in INSTRUCTION


class TestVariantRows:
    @pytest.mark.parametrize(
        "line",
        [
            "Net income including noncontrolling interests | $ | 7,367",
            "Net earnings from continuing operations | 7,367",
            "Pretax segment operating income | 7,367",
        ],
    )
    def test_a_row_naming_a_variant_is_dropped(self, line):
        concept = "operating income" if "operating" in line else "net income"
        labels, dropped, _ = label_passage(line, {(concept, 2024): [7_367_000_000.0]})
        assert labels == [] and len(dropped) == 1

    def test_a_heading_above_does_not_make_a_total_a_variant(self):
        cands = {("total revenue", 2024): [60_922_000_000.0]}
        passage = "Segment information\nYear Ended Jan 28, 2024\nTotal revenue | $ | 60,922"
        labels, _, _ = label_passage(passage, cands)
        assert len(labels) == 1
