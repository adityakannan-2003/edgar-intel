"""OpenAI-compatible chat + embeddings client.

Deliberately written against the wire format rather than a vendor SDK, so the
same code points at OpenAI, vLLM, Ollama, LM Studio, Together, or anything else
that speaks the same endpoints. Swapping providers is a base-URL change, which
is what makes the cost/quality comparison in the fine-tuning benchmark possible
at all.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
from ..config import get_settings
from ..retry import retry
from .base import Completion


class RetryableHTTPError(Exception):
    pass


def _raise_for_retry(resp: httpx.Response) -> None:
    # 429 and 5xx are worth retrying; 4xx otherwise means the request is wrong
    # and retrying just burns the rate limit.
    if resp.status_code == 429 or resp.status_code >= 500:
        raise RetryableHTTPError(f"{resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()


class OpenAICompatLLM:
    name = "openai"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        s = get_settings()
        self.base_url = (base_url or s.llm_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else s.llm_api_key
        self.model = model or s.llm_model
        self.timeout = timeout or s.llm_timeout_s
        self._client = httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    @retry(exceptions=(RetryableHTTPError, httpx.TransportError), attempts=4)
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = self._client.post(f"{self.base_url}{path}", json=payload, headers=headers)
        _raise_for_retry(resp)
        return resp.json()

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            # Structured outputs. Falls back to json_object mode on servers that
            # do not implement the schema variant.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "response"),
                    "schema": json_schema,
                    "strict": True,
                },
            }

        started = time.perf_counter()
        try:
            data = self._post("/chat/completions", payload)
        except httpx.HTTPStatusError as exc:
            if json_schema is not None and exc.response.status_code == 400:
                payload["response_format"] = {"type": "json_object"}
                data = self._post("/chat/completions", payload)
            else:
                raise
        latency_ms = int((time.perf_counter() - started) * 1000)

        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            model=data.get("model", payload["model"]),
            raw=data,
        )


class OpenAICompatEmbedder:
    name = "api"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        dim: int | None = None,
    ) -> None:
        s = get_settings()
        self.base_url = (base_url or s.llm_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else s.llm_api_key
        self.model = model or s.embed_model
        self.dim = dim or s.embed_dim
        self._client = httpx.Client(timeout=s.llm_timeout_s)

    @retry(exceptions=(RetryableHTTPError, httpx.TransportError), attempts=4)
    def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = self._client.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            headers=headers,
        )
        _raise_for_retry(resp)
        data = resp.json()
        return [row["embedding"] for row in sorted(data["data"], key=lambda r: r["index"])]


def parse_json_strict(text: str) -> dict[str, Any]:
    """Parse a model's JSON output, tolerating fenced code blocks.

    Models wrap JSON in ```json fences often enough that handling it here is
    cheaper than an extra retry, but anything beyond that is a real failure and
    should raise rather than be silently patched up.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    return json.loads(cleaned)
