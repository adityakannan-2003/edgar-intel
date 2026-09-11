"""Deterministic providers for tests and CI.

These are not mocks that return a fixed string. The fake LLM actually answers
the numeric questions correctly by reading the context it is given, and the
fake embedder produces a stable hashed vector with real lexical overlap
signal. That means the CI eval gate exercises the whole pipeline -- chunking,
retrieval, fusion, scoring, gating -- and only the model weights are stubbed.

A gate that passes because everything is mocked is theatre. This one fails if
retrieval breaks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Any

from .base import Completion

_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class FakeLLM:
    """Answers from context by simple extraction. No network, no randomness."""

    name = "fake"

    def __init__(self, model: str = "fake-1") -> None:
        self.model = model
        self.calls = 0

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
        started = time.perf_counter()
        self.calls += 1
        text = self._respond(prompt, json_schema)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return Completion(
            text=text,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(text) // 4),
            latency_ms=latency_ms,
            model=model or self.model,
        )

    # -- internals ---------------------------------------------------------
    def _respond(self, prompt: str, json_schema: dict[str, Any] | None) -> str:
        if json_schema is not None:
            return json.dumps(self._structured(prompt, json_schema))
        return self._extract_answer(prompt)

    def _structured(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        props: dict[str, Any] = schema.get("properties", {})
        out: dict[str, Any] = {}
        for key, spec in props.items():
            kind = spec.get("type", "string")
            if key in {"verdict", "correct", "passed"}:
                out[key] = self._judge_verdict(prompt)
            elif key in {"confidence", "score"}:
                out[key] = 0.9
            elif key == "rationale":
                out[key] = "deterministic fake judge: compared normalised answer to expected"
            elif key == "answer":
                out[key] = self._extract_answer(prompt)
            elif kind == "number" or kind == "integer":
                out[key] = 0
            elif kind == "boolean":
                out[key] = True
            elif kind == "array":
                out[key] = []
            elif kind == "object":
                out[key] = {}
            else:
                out[key] = self._extract_answer(prompt)
        return out

    def _judge_verdict(self, prompt: str) -> bool:
        """Grade by token overlap between the CANDIDATE and REFERENCE blocks."""
        cand = _section(prompt, "CANDIDATE")
        ref = _section(prompt, "REFERENCE")
        if not cand or not ref:
            return True
        a, b = set(_tokens(cand)), set(_tokens(ref))
        if not b:
            return True
        return len(a & b) / len(b) >= 0.5

    def _extract_answer(self, prompt: str) -> str:
        """Pull the first number out of the supplied context, else echo a span.

        Crude by design: the point is that the answer depends on whether
        retrieval actually put the right passage in front of it.
        """
        context = _section(prompt, "CONTEXT") or prompt
        nums = _NUM.findall(context)
        if nums:
            return nums[0].replace("$", "").replace(",", "")
        return context.strip()[:200]


class FakeEmbedder:
    """Hashed bag-of-words embedding.

    Deterministic, dependency-free, and -- crucially -- documents that share
    vocabulary land near each other, so cosine similarity carries real signal
    and retrieval tests are meaningful rather than vacuous.
    """

    name = "fake"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = _tokens(text)
        if not toks:
            vec[0] = 1.0
            return vec
        for tok in toks:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FakeReranker:
    """Token-overlap reranker. Stands in for a cross-encoder in CI."""

    name = "fake"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(_tokens(query))
        if not q:
            return [0.0] * len(passages)
        out = []
        for p in passages:
            toks = set(_tokens(p))
            out.append(len(q & toks) / len(q))
        return out


def _section(prompt: str, header: str) -> str:
    """Extract a block delimited by `HEADER:` ... up to the next ALL-CAPS header."""
    pattern = rf"{header}:\s*(.*?)(?=\n[A-Z][A-Z _]{{2,}}:|\Z)"
    m = re.search(pattern, prompt, flags=re.S)
    return m.group(1).strip() if m else ""
