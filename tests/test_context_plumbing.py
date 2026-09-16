"""The item-boost plumbing defect, and the character cap it was hiding.

The probe made two retrieval calls per arm: a diagnostic `search()` for the
pre-truncation ranks, and `answer_question()` for the answer. Only the first
received `item_boost`. So `--item-boost 2.0` reported ranks improving from
[9, 11, 13, 21, 29] to [3, 4, 11, 13, 15] while the answer was still generated
from the unboosted ordering. Every arm of that experiment was void, and nothing
errored.

The guard is structural, not a bigger test: one `search()` call now returns both
views, so they cannot be configured differently. These tests hold that property
in place, and pin the character-cap behaviour that the same experiment exposed.
"""

from __future__ import annotations

import pytest

from edgar_intel.retrieval.search import (
    DEFAULT_CONTEXT_MAX_CHARS,
    Hit,
    RetrievalResult,
    build_context,
    build_context_report,
    item_ranking,
    reciprocal_rank_fusion,
)


def hit(chunk_id: str, item: str | None = None, rank: int = 1,
        source: str = "dense", body: str = "x" * 1900) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        body=body,
        filing_id=1,
        score=1.0,
        source=source,
        cik="0000320193",
        ticker="AAPL",
        fiscal_year=2025,
        item=item,
        ranks={source: rank},
    )


class TestOneCallTwoViews:
    """`candidates` and `hits` must come from the same retrieval, by construction."""

    def test_hits_are_candidates_truncated(self):
        candidates = [hit(str(i)) for i in range(20)]
        result = RetrievalResult(
            hits=candidates[:8], latency_ms=1, candidates=candidates
        )
        assert result.ids == result.candidate_ids[:8]

    def test_ranks_of_reports_both_views(self):
        candidates = [hit(str(i)) for i in range(20)]
        result = RetrievalResult(hits=candidates[:8], latency_ms=1, candidates=candidates)
        relevant = {"2", "9", "11"}
        assert result.ranks_of(relevant, pre_truncation=True) == [3, 10, 12]
        # Only "2" survives top_n=8.
        assert result.ranks_of(relevant) == [3]

    def test_probe_makes_exactly_one_retrieval_call_per_arm(self):
        """The regression itself, checked on the parse tree rather than the text.

        Counting a substring would also match the docstring that explains the
        bug, which is precisely the kind of false signal that lets a real
        regression through.
        """
        import ast
        import inspect

        from edgar_intel.evals import context_probe

        tree = ast.parse(inspect.getsource(context_probe))
        called = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        assert called.count("answer_question") == 1
        # A bare search() in the probe means a second, separately configured
        # retrieval -- the exact shape of the defect.
        assert called.count("search") == 0

    def test_answer_question_accepts_every_knob_the_probe_varies(self):
        """A knob the probe can set but the answering path cannot is the bug."""
        import inspect

        from edgar_intel.evals.runner import answer_question

        params = inspect.signature(answer_question).parameters
        for knob in ("item_boost", "context_max_chars", "context_packing", "top_n"):
            assert knob in params, f"answer_question cannot be given {knob}"


class TestItemBoostChangesRanking:
    def test_zero_boost_is_identical_to_the_baseline(self):
        """The default must be a provable no-op or every A/B is void."""
        dense = [hit("boiler", "1A", 1), hit("legal", "3", 2)]
        lexical = [hit("boiler", "1A", 1, "lexical"), hit("legal", "3", 2, "lexical")]

        baseline = [h.chunk_id for h in reciprocal_rank_fusion([dense, lexical])]
        scoped = item_ranking(reciprocal_rank_fusion([dense, lexical]), ["3"])
        zeroed = [
            h.chunk_id
            for h in reciprocal_rank_fusion([dense, lexical, scoped], weights=[1.0, 1.0, 0.0])
        ]
        assert zeroed == baseline

    def test_positive_boost_lifts_item_3_for_a_legal_query(self):
        from edgar_intel.retrieval.search import infer_expected_items

        items = infer_expected_items("What material legal proceedings does Apple disclose?")
        dense = [hit("boiler", "1A", 1), hit("dma", "3", 2), hit("doj", "3", 3)]
        lexical = [hit("boiler", "1A", 1, "lexical"), hit("dma", "3", 2, "lexical")]

        before = reciprocal_rank_fusion([dense, lexical])
        assert before[0].chunk_id == "boiler"

        scoped = item_ranking(before, items)
        after = reciprocal_rank_fusion([dense, lexical, scoped], weights=[1.0, 1.0, 2.0])
        assert after[0].chunk_id == "dma"
        assert [h.chunk_id for h in after[:2]] == ["dma", "doj"]


class TestContextCap:
    """`result.hits` is not the context the model saw.

    Prompt tokens sat flat near 3,000 whether top_n was 8 or 20. That is the cap
    binding at the same place every time -- the extra passages were selected,
    counted, reported, and never sent.
    """

    def test_cap_binds_and_is_reported(self):
        # 1,900-char bodies plus a tag; six fit in 12,000 chars, the rest do not.
        report = build_context_report([hit(str(i)) for i in range(20)])
        assert report.cap_binding
        assert report.n_selected == 20
        assert report.n_included < 20
        assert report.chars_after_cap <= DEFAULT_CONTEXT_MAX_CHARS
        assert report.chars_before_cap > DEFAULT_CONTEXT_MAX_CHARS

    def test_included_count_is_flat_across_top_n_once_the_cap_binds(self):
        """Why raising top_n bought nothing: the prompt stops in the same place."""
        counts = {
            n: build_context_report([hit(str(i)) for i in range(n)]).n_included
            for n in (8, 12, 16, 20)
        }
        assert len(set(counts.values())) == 1, counts

    def test_every_dropped_passage_is_accounted_for(self):
        report = build_context_report([hit(str(i)) for i in range(20)])
        assert len(report.included_ids) + len(report.dropped_ids) == report.n_selected
        assert set(report.included_ids).isdisjoint(report.dropped_ids)

    def test_first_excluded_hit_is_named(self):
        report = build_context_report([hit(str(i)) for i in range(20)])
        assert report.first_excluded_id is not None
        assert report.first_excluded_rank == report.n_included + 1
        assert report.first_excluded_chars > 0

    def test_relevant_chunks_at_ranks_3_and_4_do_reach_the_prompt(self):
        """Instruction 9's prediction, pinned.

        With the boost, the relevant chunks land at pre-truncation ranks 3 and 4.
        Those are inside the cap, so they must appear in the prompt -- if they
        vanish, the cause is somewhere this test does not cover and needs
        finding, not explaining away.
        """
        hits = [hit(str(i)) for i in range(8)]
        report = build_context_report(hits)
        assert "2" in report.included_ids  # rank 3
        assert "3" in report.included_ids  # rank 4

    def test_greedy_stop_drops_passages_that_would_have_fit(self):
        """The sharp edge in the shipped policy: one long passage ends the loop."""
        hits = [
            hit("small-1", body="a" * 100),
            hit("huge", body="b" * 11_950),
            hit("small-2", body="c" * 100),
        ]
        greedy = build_context_report(hits, packing="greedy-stop")
        assert greedy.included_ids == ["small-1"]
        assert "small-2" in greedy.dropped_ids
        assert greedy.would_fit_with_skip == 2

    def test_skip_oversized_keeps_the_later_short_passage(self):
        hits = [
            hit("small-1", body="a" * 100),
            hit("huge", body="b" * 11_950),
            hit("small-2", body="c" * 100),
        ]
        skipped = build_context_report(hits, packing="skip-oversized")
        assert skipped.included_ids == ["small-1", "small-2"]
        assert skipped.dropped_ids == ["huge"]

    def test_default_packing_is_unchanged(self):
        """Measure before changing what the model reads."""
        import inspect

        from edgar_intel.retrieval.search import build_context_report as fn

        assert inspect.signature(fn).parameters["packing"].default == "greedy-stop"

    def test_build_context_still_returns_a_string(self):
        """Existing callers (serving, agent) must be untouched."""
        text = build_context([hit("a", body="hello")])
        assert isinstance(text, str)
        assert "hello" in text

    def test_no_cap_pressure_includes_everything(self):
        report = build_context_report([hit(str(i), body="short") for i in range(5)])
        assert not report.cap_binding
        assert report.n_included == 5
        assert report.dropped_ids == []


class TestRerankNoLongerTruncates:
    """Two places that shorten a list are two places to look when one vanishes."""

    def test_rerank_signature_has_no_top_n(self):
        import inspect

        from edgar_intel.retrieval.search import rerank

        assert "top_n" not in inspect.signature(rerank).parameters

    def test_empty_input(self):
        from edgar_intel.retrieval.search import rerank

        assert rerank("q", []) == []


class TestRecommendationSeparatesCapFromTopN:
    def _arm(self, **kw):
        from edgar_intel.evals.context_probe import ProbeArm

        base = dict(
            case_id="nar-AAPL-legal", top_n=8, use_rerank=False, git_sha="abc",
            strategy="section_aware", mode="hybrid", k=50, item_boost=2.0,
            n_relevant=5,
        )
        base.update(kw)
        return ProbeArm(**base)

    def test_cap_loss_is_named_as_distinct_from_truncation(self):
        from edgar_intel.evals.context_probe import recommend

        arm = self._arm(
            relevant_pre_truncation_ranks=[3, 4, 11, 13, 15],
            relevant_final_ranks=[3, 4],
            relevant_in_context=2,
            selected_ids=[str(i) for i in range(8)],
            included_ids=["0", "1"],
            dropped_ids=["2", "3", "4", "5", "6", "7"],
            n_included=2,
            cap_binding=True,
            relevant_ids_lost_to_cap=["2", "3"],
            relevant_in_prompt=0,
            context_chars_after_cap=11800,
            context_chars_before_cap=16000,
        )
        text = recommend([arm])
        assert "CHARACTER CAP" in text
        assert "Raising top_n cannot fix this" in text

    def test_flat_included_count_is_called_out(self):
        from edgar_intel.evals.context_probe import recommend

        arms = [
            self._arm(
                top_n=n,
                selected_ids=[str(i) for i in range(n)],
                included_ids=["0"] * 6,
                dropped_ids=["x"] * (n - 6),
                n_included=6,
                cap_binding=True,
                relevant_pre_truncation_ranks=[9],
                context_chars_after_cap=11800,
                context_chars_before_cap=n * 1900,
            )
            for n in (8, 12, 16, 20)
        ]
        assert "raising top_n changes nothing downstream" in recommend(arms)

    def test_silent_when_the_cap_never_binds(self):
        from edgar_intel.evals.context_probe import recommend

        arm = self._arm(
            passed=True,
            relevant_pre_truncation_ranks=[1, 2],
            relevant_final_ranks=[1, 2],
            relevant_in_context=2,
            relevant_in_prompt=2,
            selected_ids=["0", "1"],
            included_ids=["0", "1"],
            n_included=2,
        )
        text = recommend([arm])
        assert "CHARACTER CAP" not in text
        assert "cap binds" not in text
