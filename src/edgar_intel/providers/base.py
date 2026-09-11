"""Provider interfaces.

Everything that costs money or needs the network sits behind one of these two
protocols. That is what makes the `fake` provider possible, and the fake
provider is what makes CI able to run the full evaluation suite on every pull
request without a key and without spend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class Completion:
    """One model response, with everything needed for cost and latency books."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMProvider(Protocol):
    name: str

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
        """Return a completion. When `json_schema` is given the provider must
        make a best effort at constrained decoding and the text must parse as
        JSON conforming to that schema."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance score per passage; higher is better. Not normalised."""
        ...
