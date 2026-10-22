"""MCP server exposing the same tools the agent uses.

The point of doing this is not that MCP is fashionable. It is that the tool
definitions in tools.py become the single source of truth for three consumers
-- the internal agent loop, the HTTP API, and any MCP client (Claude Desktop,
an IDE, another agent) -- instead of being reimplemented once per surface.

That is a real architectural claim you can defend: one registry, one set of
Pydantic schemas, three transports. Adding a tool means editing one file.

Two ways to reach it:
  - hosted: Streamable HTTP at /mcp on the deployed API, so a client needs only
    the URL and no database of its own (`serving/app.py` mounts it);
  - local:  `edgar-intel mcp serve` over stdio, against your own database.

Written against the `mcp` 2.x SDK, which registers handlers in the `Server`
constructor; the 1.x decorators no longer exist.
"""

from __future__ import annotations

import json
from typing import Any

from ..obs.tracing import span
from .tools import TOOLS, ToolResult, call_tool

# `escalate_to_human` is control flow for the internal agent loop. To an outside
# client it would only echo its own arguments back, so it is not offered.
PUBLIC_TOOLS: dict[str, dict[str, Any]] = {
    name: spec for name, spec in TOOLS.items() if name != "escalate_to_human"
}


def build_server():
    try:
        import anyio
        import mcp_types as types
        from mcp.server import Server
    except ImportError as exc:
        raise ImportError(
            "MCP support needs the 'mcp' extra: pip install -e '.[mcp]'"
        ) from exc

    async def list_tools(_ctx, _params) -> types.ListToolsResult:
        # Schemas are generated from the same Pydantic models the agent
        # validates against, so an MCP client and the internal loop can never
        # drift apart.
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=name,
                    description=spec["description"],
                    input_schema=spec["args"].model_json_schema(),
                )
                for name, spec in PUBLIC_TOOLS.items()
            ]
        )

    async def dispatch(_ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        if params.name not in PUBLIC_TOOLS:
            result = ToolResult(
                ok=False,
                summary=f"Unknown tool '{params.name}'. Available: {', '.join(PUBLIC_TOOLS)}",
            )
        else:
            with span("mcp.call_tool", tool=params.name) as bag:
                # The tools are blocking Postgres reads; run off the event loop so
                # one slow query does not stall every other client of the API.
                result = await anyio.to_thread.run_sync(
                    call_tool, params.name, params.arguments or {}
                )
                bag["ok"] = result.ok
        payload = {
            "ok": result.ok,
            "summary": result.summary,
            "data": result.data,
            "citations": result.citations,
        }
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, default=str))],
            is_error=not result.ok,
        )

    return Server(
        "edgar-intel",
        instructions=(
            "SEC 10-K filings for a fixed set of companies. Use get_fact, compare_fact "
            "and compute_ratio for any number (XBRL, authoritative); search_filings for "
            "narrative text; list_coverage to see which companies and years exist."
        ),
        on_list_tools=list_tools,
        on_call_tool=dispatch,
    )


def http_session_manager():
    """Streamable HTTP for the hosted endpoint.

    Stateless with JSON responses: every request stands alone, so there are no
    sessions to lose when the machine restarts, and nothing to make sticky if a
    second machine is ever added.
    """
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    return StreamableHTTPSessionManager(
        app=build_server(), stateless=True, json_response=True
    )


def serve_stdio() -> None:  # pragma: no cover
    import asyncio

    from mcp.server.stdio import stdio_server

    server = build_server()

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_run())


def client_config(command: str = "edgar-intel", cwd: str = ".") -> dict[str, Any]:
    """Config block to paste into an MCP client's settings."""
    return {
        "mcpServers": {
            "edgar-intel": {
                "command": command,
                "args": ["mcp", "serve"],
                "cwd": cwd,
                "env": {"EDGAR_DB_DSN": "postgresql://edgar:edgar@localhost:5433/edgar"},
            }
        }
    }
