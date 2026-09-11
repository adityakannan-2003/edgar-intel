"""MCP server exposing the same tools the agent uses.

The point of doing this is not that MCP is fashionable. It is that the tool
definitions in tools.py become the single source of truth for three consumers
-- the internal agent loop, the HTTP API, and any MCP client (Claude Desktop,
an IDE, another agent) -- instead of being reimplemented once per surface.

That is a real architectural claim you can defend: one registry, one set of
Pydantic schemas, three transports. Adding a tool means editing one file.

Run it with:  edgar-intel mcp serve
"""

from __future__ import annotations

import json
from typing import Any

from .tools import TOOLS, call_tool


def build_server():  # pragma: no cover - requires the optional mcp extra
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as exc:
        raise ImportError(
            "MCP support needs the 'mcp' extra: pip install -e '.[mcp]'"
        ) from exc

    server = Server("edgar-intel")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        # Schemas are generated from the same Pydantic models the agent
        # validates against, so an MCP client and the internal loop can never
        # drift apart.
        return [
            Tool(
                name=name,
                description=spec["description"],
                inputSchema=spec["args"].model_json_schema(),
            )
            for name, spec in TOOLS.items()
        ]

    @server.call_tool()
    async def dispatch(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        result = call_tool(name, arguments or {})
        payload = {
            "ok": result.ok,
            "summary": result.summary,
            "data": result.data,
            "citations": result.citations,
        }
        return [TextContent(type="text", text=json.dumps(payload, default=str))]

    return server


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
