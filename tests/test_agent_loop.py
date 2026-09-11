"""The bounded agent loop.

Every bound is tested by making the model misbehave in the specific way that
bound exists to contain. A limit nobody has seen fire is a limit nobody knows
works.
"""

from __future__ import annotations

import json

import pytest

from edgar_intel.agent import loop as agent_loop
from edgar_intel.agent.tools import ToolResult
from edgar_intel.providers.base import Completion

# A realistic question. The input screener rejects anything under three
# characters, so a one-letter placeholder would test the screener rather
# than the loop.
QUESTION = "What were total net sales in fiscal 2023?"


class ScriptedLLM:
    """Replays a fixed list of replies, then repeats the last one forever."""

    name = "scripted"

    def __init__(self, replies: list[dict | str]) -> None:
        self.replies = replies
        self.calls = 0

    def complete(self, prompt, *, system=None, model=None, temperature=0.0,
                 max_tokens=1024, json_schema=None) -> Completion:
        idx = min(self.calls, len(self.replies) - 1)
        self.calls += 1
        reply = self.replies[idx]
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return Completion(text=text, prompt_tokens=10, completion_tokens=5, model="scripted")


@pytest.fixture
def patch_agent(monkeypatch):
    """Swap in a scripted model and a stub tool registry, with no database."""

    def _apply(replies, tool_results=None):
        llm = ScriptedLLM(replies)
        monkeypatch.setattr(agent_loop, "get_llm", lambda: llm)
        monkeypatch.setattr(agent_loop, "_persist", lambda run: None)

        results = tool_results or {}

        def fake_call_tool(name, args):
            return results.get(
                name,
                ToolResult(ok=True, summary="stub output 383,285", data={}, citations=["c1"]),
            )

        monkeypatch.setattr(agent_loop, "call_tool", fake_call_tool)
        monkeypatch.setattr(agent_loop, "TOOLS", {
            "search_filings": {}, "get_fact": {}, "escalate_to_human": {},
        })
        return llm

    return _apply


class TestHappyPath:
    def test_tool_then_answer(self, patch_agent):
        patch_agent([
            {"thought": "look it up", "tool": "get_fact", "args": {"ticker": "AAPL"}},
            {"thought": "found it", "answer": "Net sales were 383,285 million.",
             "citations": ["c1"], "confidence": 0.9},
        ])
        run = agent_loop.run_agent(QUESTION, persist=False)
        assert run.outcome == "answered"
        assert "383,285" in run.answer
        assert run.citations == ["c1"]
        assert len(run.steps) == 2

    def test_cost_is_accumulated(self, patch_agent):
        patch_agent([{"thought": "done", "answer": "A narrative answer.",
                      "citations": [], "confidence": 0.9}])
        run = agent_loop.run_agent("What risks are disclosed?", persist=False)
        assert run.prompt_tokens > 0
        assert run.total_cost_usd >= 0


class TestStepCeiling:
    def test_never_exceeds_max_steps(self, patch_agent):
        """A model that only ever calls tools must terminate, not run forever."""
        patch_agent([
            {"thought": "again", "tool": "search_filings", "args": {"query": "x"}},
        ])
        run = agent_loop.run_agent(QUESTION, max_steps=4, persist=False)
        assert run.outcome == "exhausted"
        assert len(run.steps) <= 4

    def test_exhausted_run_does_not_invent_an_answer(self, patch_agent):
        patch_agent([{"thought": "loop", "tool": "search_filings", "args": {"query": "x"}}])
        run = agent_loop.run_agent(QUESTION, max_steps=3, persist=False)
        assert run.confidence == 0.0
        assert "No confident answer" in run.answer


class TestLoopDetection:
    def test_identical_calls_are_refused(self, patch_agent):
        patch_agent([
            {"thought": "same again", "tool": "get_fact", "args": {"ticker": "AAPL"}},
        ])
        run = agent_loop.run_agent(QUESTION, max_steps=6, persist=False)
        notes = " ".join(s.note for s in run.steps)
        assert "refusing to repeat" in notes


class TestRetryBudget:
    def test_malformed_json_is_retried_then_fails_cleanly(self, patch_agent):
        patch_agent(["this is not json at all"])
        run = agent_loop.run_agent(QUESTION, max_retries=2, max_steps=8, persist=False)
        assert run.outcome == "failed"
        assert "valid JSON" in run.answer

    def test_recovers_when_the_model_corrects_itself(self, patch_agent):
        patch_agent([
            "not json",
            {"thought": "ok now", "answer": "A proper answer.",
             "citations": [], "confidence": 0.95},
        ])
        run = agent_loop.run_agent(QUESTION, max_retries=2, persist=False)
        assert run.outcome == "answered"


class TestConfidenceFloor:
    def test_low_confidence_escalates_rather_than_answering(self, patch_agent):
        patch_agent([
            {"thought": "not sure", "answer": "Possibly something.",
             "citations": [], "confidence": 0.2},
        ])
        run = agent_loop.run_agent(QUESTION, confidence_floor=0.55, persist=False)
        assert run.outcome == "escalated"
        assert "Low confidence" in run.answer

    def test_above_floor_answers(self, patch_agent):
        patch_agent([
            {"thought": "sure", "answer": "Confident answer.",
             "citations": [], "confidence": 0.8},
        ])
        assert agent_loop.run_agent(QUESTION, confidence_floor=0.55, persist=False).outcome == "answered"


class TestGroundingEnforcement:
    def test_invented_citation_is_rejected(self, patch_agent):
        patch_agent([
            {"thought": "answer", "answer": "Something.",
             "citations": ["not-a-real-id"], "confidence": 0.9},
        ])
        run = agent_loop.run_agent(QUESTION, max_retries=0, persist=False)
        assert run.outcome == "escalated"
        assert "never returned by any tool" in run.answer

    def test_fabricated_figure_is_rejected(self, patch_agent):
        """The hallucination check that runs in code, not in another model."""
        patch_agent([
            {"thought": "look", "tool": "get_fact", "args": {"ticker": "AAPL"}},
            {"thought": "answer", "answer": "Revenue was 999,999,999.",
             "citations": ["c1"], "confidence": 0.99},
        ])
        run = agent_loop.run_agent(QUESTION, max_retries=0, persist=False)
        assert run.outcome == "escalated"
        assert "not present in any tool output" in run.answer


class TestEscalation:
    def test_escalate_tool_ends_the_run(self, patch_agent):
        patch_agent(
            [{"thought": "cannot do this", "tool": "escalate_to_human",
              "args": {"reason": "no coverage for that company"}}],
            tool_results={
                "escalate_to_human": ToolResult(
                    ok=True, summary="ESCALATED: no coverage for that company"
                )
            },
        )
        run = agent_loop.run_agent(QUESTION, persist=False)
        assert run.outcome == "escalated"
        assert "no coverage" in run.answer


class TestUnknownTool:
    def test_unknown_tool_is_reported_not_crashed(self, patch_agent):
        patch_agent([
            {"thought": "try", "tool": "does_not_exist", "args": {}},
            {"thought": "ok", "answer": "Recovered answer.",
             "citations": [], "confidence": 0.9},
        ])
        run = agent_loop.run_agent(QUESTION, persist=False)
        assert run.outcome == "answered"
        assert any("unknown tool" in s.note for s in run.steps)
