"""The structured-output request each caller sends, against a stand-in for OpenAI.

The stand-in applies the strict-mode rules OpenAI enforces with HTTP 400: every
object must set `additionalProperties: false` and list all its properties in
`required`. Until this was pinned, every agent step was rejected in ~250 ms and
then re-sent as json_object.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from edgar_intel.agent.loop import ACTION_SCHEMA
from edgar_intel.evals.schemas import ANSWER_SCHEMA, JUDGE_SCHEMA
from edgar_intel.providers.openai_compat import OpenAICompatLLM


def _strict_ok(schema: dict[str, Any]) -> bool:
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            return False
        if set(schema.get("required", [])) != set(props):
            return False
        return all(_strict_ok(p) for p in props.values())
    if schema.get("type") == "array":
        return _strict_ok(schema.get("items", {}))
    return True


def _llm(handler) -> OpenAICompatLLM:
    llm = OpenAICompatLLM(base_url="http://llm.test/v1", api_key="test", model="m")
    llm._client = httpx.Client(transport=httpx.MockTransport(handler))
    return llm


def _openai_like(sent: list[dict[str, Any]]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body["response_format"])
        fmt = body["response_format"]
        if fmt["type"] == "json_schema" and not _strict_ok(fmt["json_schema"]["schema"]):
            return httpx.Response(400, json={"error": {"message": "Invalid schema"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    return handler


def test_the_agent_action_is_accepted_on_the_first_request():
    sent: list[dict[str, Any]] = []
    _llm(_openai_like(sent)).complete("Next action (JSON only):", json_schema=ACTION_SCHEMA)
    assert sent == [{"type": "json_object"}]


@pytest.mark.parametrize("schema", [JUDGE_SCHEMA, ANSWER_SCHEMA], ids=lambda s: s["title"])
def test_closed_schemas_go_strict_and_are_accepted_first_time(schema):
    sent: list[dict[str, Any]] = []
    _llm(_openai_like(sent)).complete("JSON please", json_schema=schema)
    assert len(sent) == 1
    assert sent[0]["type"] == "json_schema" and sent[0]["json_schema"]["strict"] is True


def test_a_server_without_json_schema_still_gets_json_object():
    sent: list[dict[str, Any]] = []

    def no_schema_support(request: httpx.Request) -> httpx.Response:
        fmt = json.loads(request.content)["response_format"]
        sent.append(fmt)
        if fmt["type"] == "json_schema":
            return httpx.Response(400, json={"error": {"message": "unsupported"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    _llm(no_schema_support).complete("JSON please", json_schema=JUDGE_SCHEMA)
    assert [f["type"] for f in sent] == ["json_schema", "json_object"]
