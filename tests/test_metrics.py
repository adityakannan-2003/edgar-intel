"""Metric definitions.

These tests exist because a subtly wrong metric is worse than no metric: it
produces a number that looks authoritative and is wrong, and nobody checks it
again. Each case below is hand-computed.
"""

from __future__ import annotations

import math

import pytest

from edgar_intel.retrieval.metrics import (
    aggregate,
    cohens_kappa,
    dcg_at_k,
    hit_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarise,
)


class TestRecall:
    def test_all_relevant_in_top_k(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_partial(self):
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5

    def test_k_truncates(self):
        # 'b' sits at rank 3, so recall@2 must not see it.
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5

    def test_empty_relevant_returns_zero_not_error(self):
        assert recall_at_k(["a"], set(), 5) == 0.0

    def test_no_hits(self):
        assert recall_at_k(["x", "y"], {"a"}, 2) == 0.0


class TestPrecision:
    def test_half(self):
        assert precision_at_k(["a", "x"], {"a"}, 2) == 0.5

    def test_denominator_is_retrieved_count_not_k(self):
        # Only one document was retrieved; precision@5 is 1/1, not 1/5.
        assert precision_at_k(["a"], {"a"}, 5) == 1.0

    def test_zero_k(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestReciprocalRank:
    @pytest.mark.parametrize(
        "ranked,expected",
        [
            (["a", "x", "y"], 1.0),
            (["x", "a", "y"], 0.5),
            (["x", "y", "a"], 1 / 3),
            (["x", "y", "z"], 0.0),
        ],
    )
    def test_rank_positions(self, ranked, expected):
        assert reciprocal_rank(ranked, {"a"}) == pytest.approx(expected)

    def test_mrr_averages(self):
        runs = [(["a"], {"a"}), (["x", "a"], {"a"})]
        assert mean_reciprocal_rank(runs) == pytest.approx(0.75)

    def test_mrr_empty(self):
        assert mean_reciprocal_rank([]) == 0.0


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)

    def test_worse_ranking_scores_lower(self):
        good = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
        bad = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        assert good > bad

    def test_dcg_discount_matches_formula(self):
        # gains [1, 1] -> 1/log2(2) + 1/log2(3) = 1 + 0.6309
        assert dcg_at_k([1.0, 1.0], 2) == pytest.approx(1 + 1 / math.log2(3))

    def test_recall_is_blind_to_order_but_ndcg_is_not(self):
        """The reason both are reported.

        Reranking cannot improve recall@k when k covers the whole candidate
        set -- it only moves hits upward. A project that reports only recall
        will conclude reranking did nothing.
        """
        front = ["a", "x", "y"]
        back = ["x", "y", "a"]
        assert recall_at_k(front, {"a"}, 3) == recall_at_k(back, {"a"}, 3)
        assert ndcg_at_k(front, {"a"}, 3) > ndcg_at_k(back, {"a"}, 3)

    def test_graded_relevance(self):
        graded = {"a": 3.0, "b": 1.0}
        better = ndcg_at_k(["a", "b"], {"a", "b"}, 2, graded)
        worse = ndcg_at_k(["b", "a"], {"a", "b"}, 2, graded)
        assert better == pytest.approx(1.0)
        assert worse < better


class TestHitRateAndSummary:
    def test_hit_rate_is_binary(self):
        assert hit_rate(["x", "a"], {"a"}, 2) == 1.0
        assert hit_rate(["x", "a"], {"a"}, 1) == 0.0

    def test_summarise_shape(self):
        out = summarise(["a", "b", "c"], {"b"})
        assert out["recall@1"] == 0.0
        assert out["recall@3"] == 1.0
        assert out["mrr"] == pytest.approx(0.5)
        assert out["first_hit_rank"] == 2.0

    def test_aggregate_means_and_tolerates_missing_keys(self):
        out = aggregate([{"recall@5": 1.0, "mrr": 1.0}, {"recall@5": 0.0}])
        assert out["recall@5"] == 0.5
        assert out["mrr"] == 1.0

    def test_aggregate_empty(self):
        assert aggregate([]) == {}


class TestCohensKappa:
    def test_perfect_agreement(self):
        assert cohens_kappa([True, False, True], [True, False, True]) == 1.0

    def test_total_disagreement_is_negative(self):
        assert cohens_kappa([True, False], [False, True]) < 0

    def test_unanimous_identical_raters_returns_one_not_nan(self):
        """The degenerate case that crashes naive implementations.

        When both raters say True every time, expected agreement is 1.0 and the
        formula divides by zero. A judge that passes everything on an easy set
        hits this constantly.
        """
        assert cohens_kappa([True] * 10, [True] * 10) == 1.0

    def test_chance_agreement_is_discounted(self):
        """Raw agreement 0.5 with a skewed marginal should not read as 'fine'."""
        human = [True, True, False, False]
        judge = [True, False, True, False]
        assert cohens_kappa(human, judge) == pytest.approx(0.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            cohens_kappa([True], [True, False])

    def test_empty(self):
        assert cohens_kappa([], []) == 0.0


class TestHitRateVersusRecall:
    """Why both are reported.

    On the first real baseline run, 112 of 232 cases carried five evidence
    chunks each, because a figure repeats across a table, the MD&A prose and a
    footnote. recall@5 divides by five and reported 0.26 for retrieval that was
    finding the answer; the system only ever needed one of those passages.
    """

    def test_recall_penalises_redundant_evidence(self):
        from edgar_intel.retrieval.metrics import hit_rate, recall_at_k

        # The answer appears in five chunks; retrieval surfaced one of them.
        relevant = {"a", "b", "c", "d", "e"}
        ranked = ["a", "x", "y", "z", "w"]
        assert recall_at_k(ranked, relevant, 5) == 0.2   # looks like failure
        assert hit_rate(ranked, relevant, 5) == 1.0      # the question was answered

    def test_hit_rate_is_honest_about_a_real_miss(self):
        from edgar_intel.retrieval.metrics import hit_rate

        assert hit_rate(["x", "y"], {"a"}, 2) == 0.0

    def test_recall_is_right_when_all_evidence_is_needed(self):
        """Not a replacement: a multi-hop case genuinely needs every passage."""
        from edgar_intel.retrieval.metrics import hit_rate, recall_at_k

        relevant = {"fy2023", "fy2024"}
        ranked = ["fy2023", "noise"]
        assert hit_rate(ranked, relevant, 2) == 1.0      # too generous here
        assert recall_at_k(ranked, relevant, 2) == 0.5   # correctly partial

    def test_summarise_reports_both_and_the_evidence_count(self):
        from edgar_intel.retrieval.metrics import summarise

        out = summarise(["a", "x", "y"], {"a", "b", "c"})
        # summarise() rounds to 4dp, so compare at that resolution.
        assert out["recall@3"] == pytest.approx(1 / 3, abs=1e-4)
        assert out["hit@3"] == 1.0
        assert out["n_relevant"] == 3.0
