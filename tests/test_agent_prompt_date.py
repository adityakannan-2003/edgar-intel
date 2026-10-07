"""The agent is told today's date.

gpt-4o-mini's sense of the date is earlier than the corpus. With the tag
vocabulary in place, 21 of 29 escalations on the golden set were FY2025-26
figures it called "not yet reported", although `list_coverage` listed those
10-Ks. A rule telling it to disregard its sense of the date made that worse
(docs/METRICS.md §6). Stating the date is the alternative; a scripted model
cannot test a prior, so these pin the plumbing.
"""

from __future__ import annotations

import json

from edgar_intel.agent import loop as agent_loop
from edgar_intel.evals import agent_eval
from edgar_intel.providers.base import Completion


class Recorder:
    name = "recorder"

    def __init__(self) -> None:
        self.systems: list[str] = []

    def complete(self, prompt, *, system=None, model=None, temperature=0.0,
                 max_tokens=1024, json_schema=None) -> Completion:
        self.systems.append(system or "")
        reply = {"thought": "done", "answer": "A narrative answer.", "citations": [],
                 "confidence": 0.9}
        return Completion(text=json.dumps(reply), prompt_tokens=10, completion_tokens=5)


def test_the_system_prompt_states_todays_date(monkeypatch):
    llm = Recorder()
    monkeypatch.setattr(agent_loop, "get_llm", lambda: llm)
    monkeypatch.setattr(agent_loop, "prompt_date", lambda: "2026-10-06")
    agent_loop.run_agent("What supply risks does Apple disclose?", persist=False)
    assert "Today's date is 2026-10-06." in llm.systems[0]


def test_the_date_comes_from_the_clock():
    from datetime import date

    assert agent_loop.prompt_date() == date.today().isoformat()


def test_an_evaluation_records_the_date_it_ran_with(monkeypatch):
    monkeypatch.setattr(agent_eval, "prompt_date", lambda: "2026-10-06")
    assert agent_eval.agent_config()["agent_prompt_date"] == "2026-10-06"
