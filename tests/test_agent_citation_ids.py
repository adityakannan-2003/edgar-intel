"""`get_fact` must print the citation id it registers.

It printed "(source: XBRL 0000080424-24-000083)" and registered
"xbrl:0000080424-24-000083". gpt-4o-mini cites what it reads, so it cited the
bare accession, the citation check rejected an id no tool had returned, and the
agent escalated with the right figure in hand. In the 20-case pilot that
happened on 11 questions and ended 7 of them in escalation, two of them P&G and
UNH net income, the figures `get_fact` exists to get right.

The existing end-to-end test did not catch it because its scripted model cited
the registered form, which the author knew and the model could not.
"""

from __future__ import annotations

import json
import re

from edgar_intel.agent import loop as agent_loop
from edgar_intel.agent import tools
from edgar_intel.providers.base import Completion

PG_FY2024 = {
    "value": 14_879_000_000.0,
    "unit": "USD",
    "tag": "NetIncomeLoss",
    "fiscal_year": 2024,
    "accession": "0000080424-24-000083",
    "name": "PROCTER & GAMBLE Co",
}


def install(monkeypatch) -> None:
    monkeypatch.setattr(tools.db, "query_one", lambda sql, params: PG_FY2024)
    monkeypatch.setattr(tools.db, "query", lambda sql, params=None: [])


class ModelThatCitesWhatItReads:
    """Looks the figure up, then cites the last token of the printed source."""

    name = "copying"

    def complete(self, prompt, *, system=None, model=None, temperature=0.0,
                 max_tokens=1024, json_schema=None) -> Completion:
        printed = re.findall(r"\(source: ([^)]*)\)", prompt)
        if not printed:
            reply = {"thought": "look it up", "tool": "get_fact",
                     "args": {"ticker": "PG", "tag": "NetIncomeLoss", "fiscal_year": 2024}}
        elif "ERROR" in prompt:
            reply = {"thought": "give up", "tool": "escalate_to_human",
                     "args": {"reason": "citation rejected"}}
        else:
            reply = {"thought": "found it", "answer": "Net income was $14.88 billion.",
                     "citations": [printed[-1].split()[-1]], "confidence": 0.95}
        return Completion(text=json.dumps(reply), prompt_tokens=10, completion_tokens=5)


def test_the_printed_id_is_the_registered_id(monkeypatch):
    install(monkeypatch)
    result = tools.call_tool(
        "get_fact", {"ticker": "PG", "tag": "NetIncomeLoss", "fiscal_year": 2024}
    )
    assert result.ok, result.summary
    assert result.citations == ["xbrl:0000080424-24-000083"]
    assert f"(source: {result.citations[0]})" in result.summary


def test_a_model_citing_what_it_read_is_answered_not_escalated(monkeypatch):
    install(monkeypatch)
    monkeypatch.setattr(agent_loop, "get_llm", ModelThatCitesWhatItReads)
    run = agent_loop.run_agent(
        "How much net income did PROCTER & GAMBLE Co report in FY2024?", persist=False
    )
    assert run.outcome == "answered", run.render()
    assert run.citations == ["xbrl:0000080424-24-000083"]
    assert not [s for s in run.steps if s.kind == "error"]
