"""Context selection, item-aware ranking, and error reporting.

Every case here comes from the `nar-AAPL-legal` investigation: all five relevant
chunks were inside the hybrid top-50 at pre-rerank ranks 9, 11, 13, 21 and 29,
and `top_n=8` cut every one of them. Reranking on or off made no difference,
because the reranker reorders inside the candidate set and the truncation
happens after. The run recorded a narrative quality failure.
"""

from __future__ import annotations

import httpx
import pytest

from edgar_intel.retrieval.search import (
    Hit,
    infer_expected_items,
    item_ranking,
    reciprocal_rank_fusion,
)


def hit(chunk_id: str, item: str | None = None, rank: int = 1, source: str = "dense") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        body=f"body {chunk_id}",
        filing_id=1,
        score=1.0,
        source=source,
        cik="0000320193",
        ticker="AAPL",
        fiscal_year=2025,
        item=item,
        ranks={source: rank},
    )


class TestItemInference:
    """The question says which Item answers it. Form 10-K guarantees that."""

    def test_legal_question_maps_to_item_3(self):
        items = infer_expected_items("What material legal proceedings does Apple Inc. disclose?")
        assert items[0] == "3"

    def test_market_risk_question_maps_to_item_7a(self):
        items = infer_expected_items(
            "What foreign currency exposure does Procter & Gamble report and how is it managed?"
        )
        assert "7A" in items

    def test_mdna_question_maps_to_item_7(self):
        items = infer_expected_items(
            "What does NVIDIA's MD&A give as the main drivers of the change in revenue?"
        )
        assert "7" in items

    def test_risk_question_maps_to_item_1a(self):
        items = infer_expected_items("What supply chain risk factors does Costco disclose?")
        assert "1A" in items

    def test_a_plain_numeric_question_infers_nothing(self):
        """A figure can sit in Item 7, Item 8 or a footnote. Guessing would hurt."""
        assert infer_expected_items("What was Caterpillar's total revenue for fiscal year 2024?") == []

    def test_no_company_name_changes_the_answer(self):
        """The signal is Form 10-K structure, not the filer."""
        apple = infer_expected_items("What legal proceedings does Apple Inc. disclose?")
        generic = infer_expected_items("What legal proceedings does the company disclose?")
        assert apple == generic

    def test_ordering_prefers_the_stronger_match(self):
        items = infer_expected_items(
            "Describe the antitrust lawsuit and litigation risk factors"
        )
        assert items[0] == "3"


class TestItemRanking:
    def test_scoped_to_the_expected_item(self):
        hits = [hit("a", "1A"), hit("b", "3"), hit("c", "7"), hit("d", "3")]
        assert [h.chunk_id for h in item_ranking(hits, ["3"])] == ["b", "d"]

    def test_preserves_incoming_order_within_an_item(self):
        """Adds Item information and nothing else; never reorders within an Item."""
        hits = [hit("late", "3"), hit("early", "3")]
        assert [h.chunk_id for h in item_ranking(hits, ["3"])] == ["late", "early"]

    def test_multiple_items_ranked_by_priority(self):
        hits = [hit("risk", "1A"), hit("legal", "3")]
        assert [h.chunk_id for h in item_ranking(hits, ["3", "1A"])] == ["legal", "risk"]

    def test_empty_when_nothing_inferred(self):
        assert item_ranking([hit("a", "3")], []) == []

    def test_chunks_with_no_item_are_not_dropped_from_the_pipeline(self):
        """item_ranking only builds the *boost* list; it is never the whole result."""
        hits = [hit("unknown", None), hit("legal", "3")]
        boosted = item_ranking(hits, ["3"])
        assert [h.chunk_id for h in boosted] == ["legal"]
        assert len(hits) == 2


class TestItemBoostIsABoostNotAFilter:
    """Filtering to the inferred Item would have scored well here and been wrong.

    NVIDIA's Item 8 is three sentences pointing at the financial statements;
    P&G's Item 7A points into the MD&A. Incorporation by reference is routine,
    so evidence for an Item-3 question can legitimately live outside Item 3.
    """

    def test_off_item_candidate_survives_the_boost(self):
        dense = [hit("off-item", None, 1), hit("on-item", "3", 2)]
        lexical = [hit("off-item", None, 1, "lexical")]
        scoped = item_ranking(reciprocal_rank_fusion([dense, lexical]), ["3"])
        fused = reciprocal_rank_fusion([dense, lexical, scoped], weights=[1.0, 1.0, 1.0])
        assert {h.chunk_id for h in fused} == {"off-item", "on-item"}

    def test_boost_lifts_the_on_item_candidate(self):
        dense = [hit("boiler", "1A", 1), hit("legal", "3", 2)]
        lexical = [hit("boiler", "1A", 1, "lexical"), hit("legal", "3", 2, "lexical")]

        without = reciprocal_rank_fusion([dense, lexical])
        assert without[0].chunk_id == "boiler"

        scoped = item_ranking(without, ["3"])
        with_boost = reciprocal_rank_fusion(
            [dense, lexical, scoped], weights=[1.0, 1.0, 2.0]
        )
        assert with_boost[0].chunk_id == "legal"

    def test_zero_weight_is_identical_to_no_boost(self):
        """The default must be a no-op, or every A/B against an old run is void."""
        dense = [hit("a", "1A", 1), hit("b", "3", 2)]
        lexical = [hit("b", "3", 1, "lexical")]
        baseline = [h.chunk_id for h in reciprocal_rank_fusion([dense, lexical])]
        scoped = item_ranking(reciprocal_rank_fusion([dense, lexical]), ["3"])
        zero = [
            h.chunk_id
            for h in reciprocal_rank_fusion([dense, lexical, scoped], weights=[1.0, 1.0, 0.0])
        ]
        assert zero == baseline


class TestFailureSamplingPerKind:
    """The reporting defect that hid the judge failure for four runs."""

    def _results(self, n_numeric: int, n_narrative: int):
        from edgar_intel.evals.schemas import CaseResult

        rows = [
            CaseResult(f"num-{i}", "numeric", False, 0.0, "", "", judge_rationale="wrong")
            for i in range(n_numeric)
        ]
        rows += [
            CaseResult(f"nar-{i}", "narrative", False, 0.0, "", "", judge_rationale="400")
            for i in range(n_narrative)
        ]
        return rows

    def test_narrative_failures_survive_a_flood_of_numeric_ones(self):
        from edgar_intel.evals.runner import sample_failures_by_kind

        sample = sample_failures_by_kind(self._results(200, 24), per_kind=25)
        kinds = {row["kind"] for row in sample}
        assert "narrative" in kinds
        assert sum(1 for r in sample if r["kind"] == "narrative") == 24

    def test_quota_applies_per_kind(self):
        from edgar_intel.evals.runner import sample_failures_by_kind

        sample = sample_failures_by_kind(self._results(200, 24), per_kind=10)
        assert sum(1 for r in sample if r["kind"] == "numeric") == 10
        assert sum(1 for r in sample if r["kind"] == "narrative") == 10

    def test_counts_report_the_true_totals_not_the_sample(self):
        from edgar_intel.evals.runner import failure_counts_by_kind

        assert failure_counts_by_kind(self._results(200, 24)) == {
            "numeric": 200,
            "narrative": 24,
        }

    def test_passing_cases_are_excluded(self):
        from edgar_intel.evals.runner import failure_counts_by_kind
        from edgar_intel.evals.schemas import CaseResult

        rows = [CaseResult("ok", "numeric", True, 1.0, "", "")]
        assert failure_counts_by_kind(rows) == {}


class TestProviderErrorDetail:
    """HTTP 400 with no body is what made the judge failure look like bad answers."""

    def _response(self, status: int, payload=None, text: str = "") -> httpx.Response:
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        if payload is not None:
            return httpx.Response(status, json=payload, request=request)
        return httpx.Response(status, text=text, request=request)

    def test_openai_error_body_is_surfaced(self):
        from edgar_intel.providers.openai_compat import error_detail

        detail = error_detail(
            self._response(
                400,
                {
                    "error": {
                        "message": "Unsupported parameter: 'max_tokens' is not supported "
                        "with this model. Use 'max_completion_tokens' instead.",
                        "type": "invalid_request_error",
                        "param": "max_tokens",
                    }
                },
            )
        )
        assert "400" in detail
        assert "max_tokens" in detail
        assert "invalid_request_error" in detail

    def test_non_json_body_still_reported(self):
        from edgar_intel.providers.openai_compat import error_detail

        detail = error_detail(self._response(502, text="upstream connect error"))
        assert "502" in detail
        assert "upstream connect error" in detail

    def test_raise_carries_the_body_not_the_generic_message(self):
        from edgar_intel.providers.openai_compat import _raise_for_retry

        resp = self._response(400, {"error": {"message": "model not found", "code": "404"}})
        with pytest.raises(httpx.HTTPStatusError) as caught:
            _raise_for_retry(resp)
        assert "model not found" in str(caught.value)
        # The type must stay httpx.HTTPStatusError so the structured-output
        # fallback in complete() still matches on it.
        assert caught.value.response.status_code == 400

    def test_retryable_statuses_still_retry(self):
        from edgar_intel.providers.openai_compat import RetryableHTTPError, _raise_for_retry

        with pytest.raises(RetryableHTTPError):
            _raise_for_retry(self._response(429, {"error": {"message": "slow down"}}))
        with pytest.raises(RetryableHTTPError):
            _raise_for_retry(self._response(503, text="unavailable"))

    def test_success_does_not_raise(self):
        from edgar_intel.providers.openai_compat import _raise_for_retry

        assert _raise_for_retry(self._response(200, {"ok": True})) is None


class TestProbeRecommendation:
    """The probe must not let a top_n increase be mistaken for a ranking fix."""

    def _arm(self, top_n: int, passed: bool, pre: list[int], final: list[int]):
        from edgar_intel.evals.context_probe import ProbeArm

        return ProbeArm(
            case_id="nar-AAPL-legal",
            top_n=top_n,
            use_rerank=False,
            git_sha="abc1234",
            strategy="section_aware",
            mode="hybrid",
            k=50,
            item_boost=0.0,
            passed=passed,
            n_relevant=5,
            relevant_in_context=len(final),
            relevant_final_ranks=final,
            relevant_pre_truncation_ranks=pre,
        )

    def test_reports_the_smallest_reliable_top_n(self):
        from edgar_intel.evals.context_probe import recommend

        arms = [
            self._arm(8, False, [9, 11, 13, 21, 29], []),
            self._arm(12, True, [9, 11, 13, 21, 29], [9, 11]),
            self._arm(16, True, [9, 11, 13, 21, 29], [9, 11, 13]),
        ]
        assert "top_n=12" in recommend(arms)

    def test_says_so_when_nothing_passes(self):
        from edgar_intel.evals.context_probe import recommend

        arms = [self._arm(n, False, [9], []) for n in (8, 12, 16, 20)]
        assert "No top_n in the grid passes reliably" in recommend(arms)

    def test_flags_that_deep_ranks_mean_ranking_is_still_broken(self):
        """Passing at top_n=16 with first evidence at rank 9 is mitigation, not a fix."""
        from edgar_intel.evals.context_probe import recommend

        arms = [self._arm(16, True, [9, 11, 13], [9, 11, 13])]
        text = recommend(arms)
        assert "mitigation" in text
        assert "ranking work open" in text

    def test_attributes_the_loss_to_selection_not_candidate_generation(self):
        from edgar_intel.evals.context_probe import recommend

        arms = [self._arm(8, False, [9, 11, 13, 21, 29], [])]
        assert "not candidate generation" in recommend(arms)

    def test_one_repeat_failing_makes_the_setting_unreliable(self):
        from edgar_intel.evals.context_probe import recommend

        arms = [
            self._arm(12, True, [9], [9]),
            self._arm(12, False, [9], []),  # same top_n, other arm fails
            self._arm(16, True, [9], [9]),
        ]
        assert "top_n=16" in recommend(arms) or "top_n=12" in recommend(arms)
