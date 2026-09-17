"""The judge has to be shown what it is asked to check against.

`baseline-v4` adopted the v2 contract, whose third rule is "fail extra detail
only when it is contradicted by the supplied source context or cannot reasonably
be supported by it". `grade()` called `judge_narrative(case, answer)` with no
context, so every narrative case was graded with SOURCE CONTEXT rendered as
"(not supplied)". With no passages to check against, the only thing the judge
could compare extra detail to was the reference -- which is v1's failure mode
wearing v2's text.

`nar-COST-supply-concentration` failed on exactly that: "includes additional
risks such as labor disputes and climate change that are not mentioned in the
reference". Those risks are in Costco's Item 1A. The judge had no way to know.

This is the third instance of one shape of bug in this repo: a parameter that
reached the experiment harness and not the production path (item_boost, then the
probe's diagnostic retrieval, now source_context). The tests below are
structural for that reason -- they check the wiring, not a model's opinion.
"""

from __future__ import annotations

import ast
import inspect

from edgar_intel.evals.judge_contracts import CONTRACTS, get_contract


class TestGradePassesContextThrough:
    def test_grade_accepts_source_context(self):
        from edgar_intel.evals.judge import grade

        assert "source_context" in inspect.signature(grade).parameters

    def test_build_result_accepts_source_context(self):
        from edgar_intel.evals.judge import build_result

        assert "source_context" in inspect.signature(build_result).parameters

    def test_grade_forwards_it_to_the_judge(self):
        """The call that was missing the argument."""
        from edgar_intel.evals import judge

        tree = ast.parse(inspect.getsource(judge.grade))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "judge_narrative"
        ]
        assert len(calls) == 1
        kwargs = {kw.arg for kw in calls[0].keywords}
        assert "source_context" in kwargs, "judge_narrative called without source_context"

    def test_build_result_forwards_it_to_grade(self):
        from edgar_intel.evals import judge

        tree = ast.parse(inspect.getsource(judge.build_result))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "grade"
        ]
        assert len(calls) == 1
        passed = {kw.arg for kw in calls[0].keywords} | {
            a.id for a in calls[0].args if isinstance(a, ast.Name)
        }
        assert "source_context" in passed

    def test_run_suite_supplies_the_real_context(self):
        """Not a placeholder, not the reference -- the passages the model read."""
        from edgar_intel.evals import runner

        tree = ast.parse(inspect.getsource(runner.run_suite))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_result"
        ]
        assert len(calls) == 1
        source_kwargs = [kw for kw in calls[0].keywords if kw.arg == "source_context"]
        assert source_kwargs, "run_suite does not pass source_context"
        value = ast.unparse(source_kwargs[0].value)
        assert "text" in value, f"expected the context report's text, got {value}"


class TestConfigRecordsWhetherTheJudgeCouldSee:
    def test_run_config_records_the_flag(self):
        from edgar_intel.evals import runner

        source = inspect.getsource(runner.run_suite)
        assert "judge_sees_source_context" in source

    def test_the_flag_tracks_the_contract(self):
        assert CONTRACTS["v2"].wants_source_context
        assert not CONTRACTS["v1"].wants_source_context


class TestMissingContextIsVisibleNotSilent:
    def test_a_context_hungry_contract_marks_the_gap(self):
        """If it ever happens again it must be readable in the rendered prompt."""
        rendered = get_contract("v2").render("Q", "REF", "CAND", "")
        assert "(not supplied)" in rendered

    def test_supplied_context_reaches_the_prompt(self):
        rendered = get_contract("v2").render("Q", "REF", "CAND", "[1] AAPL Item 1A labor disputes")
        assert "labor disputes" in rendered
        assert "(not supplied)" not in rendered

    def test_v1_never_claims_to_have_context(self):
        """v1 is the control arm and must stay byte-identical."""
        rendered = get_contract("v1").render("Q", "REF", "CAND", "IGNORED")
        assert "IGNORED" not in rendered
        assert "SOURCE CONTEXT" not in rendered
