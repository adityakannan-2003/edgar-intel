"""The MCP surface: every public tool reached through a real MCP client.

Each tool's function is stubbed, so these tests need no database. What they pin
is the part that is MCP's own: the tool list matches the registry, arguments are
validated by the same Pydantic models the agent uses before a tool runs, results
come back as the JSON the agent loop sees, and failures are flagged as errors.
"""

from __future__ import annotations

import json

import anyio
import pytest

pytest.importorskip("mcp", reason="mcp extra not installed")
from mcp import Client  # noqa: E402

from edgar_intel.agent import tools  # noqa: E402
from edgar_intel.agent.mcp_server import PUBLIC_TOOLS, build_server  # noqa: E402

# One valid call per public tool. Lower-case tickers on purpose: the arg models
# upper-case them, and the stub checks the tool saw the validated value.
CALLS = {
    "search_filings": {"query": "supply chain risk", "ticker": "aapl"},
    "get_fact": {"ticker": "aapl", "tag": "revenue", "fiscal_year": 2025},
    "compare_fact": {"ticker": "aapl", "tag": "revenue", "year_a": 2024, "year_b": 2025},
    "compute_ratio": {
        "ticker": "aapl",
        "numerator_tag": "NetIncomeLoss",
        "denominator_tag": "revenue",
        "fiscal_year": 2025,
    },
    "list_coverage": {},
}


def _run(fn):
    async def main():
        async with Client(build_server()) as client:
            return await fn(client)

    return anyio.run(main)


def test_every_public_tool_has_a_call_here():
    assert set(CALLS) == set(PUBLIC_TOOLS)


def test_lists_the_registry_minus_the_internal_handoff():
    listed = _run(lambda c: c.list_tools())
    by_name = {t.name: t for t in listed.tools}
    assert set(by_name) == set(PUBLIC_TOOLS)
    assert "escalate_to_human" not in by_name
    for name, spec in PUBLIC_TOOLS.items():
        assert by_name[name].input_schema == spec["args"].model_json_schema()


@pytest.mark.parametrize("name", sorted(CALLS))
def test_each_tool_runs_with_validated_args(name, monkeypatch):
    seen = []

    def stub(args):
        seen.append(args)
        return tools.ToolResult(
            ok=True, summary=f"{name} ran", data={"n": 1}, citations=["xbrl:0000320193-25-000079"]
        )

    monkeypatch.setitem(tools.TOOLS[name], "fn", stub)
    result = _run(lambda c: c.call_tool(name, CALLS[name]))

    assert not result.is_error
    assert json.loads(result.content[0].text) == {
        "ok": True,
        "summary": f"{name} ran",
        "data": {"n": 1},
        "citations": ["xbrl:0000320193-25-000079"],
    }
    (args,) = seen
    assert isinstance(args, PUBLIC_TOOLS[name]["args"])
    if "ticker" in CALLS[name]:
        assert args.ticker == "AAPL"


def test_invalid_arguments_are_an_error_and_the_tool_never_runs(monkeypatch):
    monkeypatch.setitem(tools.TOOLS["get_fact"], "fn", lambda a: pytest.fail("ran"))
    result = _run(lambda c: c.call_tool("get_fact", {"ticker": "AAPL"}))
    assert result.is_error
    assert "Invalid arguments for get_fact" in json.loads(result.content[0].text)["summary"]


def test_a_tool_that_finds_nothing_is_flagged(monkeypatch):
    monkeypatch.setitem(
        tools.TOOLS["get_fact"], "fn", lambda a: tools.ToolResult(ok=False, summary="no such fact")
    )
    result = _run(lambda c: c.call_tool("get_fact", CALLS["get_fact"]))
    assert result.is_error
    assert json.loads(result.content[0].text)["summary"] == "no such fact"


def test_the_handoff_tool_is_not_callable_from_outside():
    result = _run(lambda c: c.call_tool("escalate_to_human", {"reason": "x"}))
    assert result.is_error
    assert "Unknown tool" in json.loads(result.content[0].text)["summary"]


def test_the_api_serves_mcp_at_slash_mcp(monkeypatch):
    pytest.importorskip("fastapi", reason="serving extra not installed")
    from fastapi.testclient import TestClient

    from edgar_intel.serving import app as app_module

    monkeypatch.setattr(app_module, "close_pool", lambda: None)
    with TestClient(app_module.app) as client:
        r = client.post(
            "/mcp",
            headers={"accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
        )
    assert r.status_code == 200, r.text
    assert r.json()["result"]["serverInfo"]["name"] == "edgar-intel"
