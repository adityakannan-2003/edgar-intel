"""OpenTelemetry tracing and token accounting.

Two reasons this is in the repo rather than bolted on later.

First, observability appears in roughly four out of five AI engineering job
descriptions, and almost nobody's portfolio project has it. Second, and more
practically: an LLM system fails in ways logs alone cannot explain. "The answer
was wrong" is not actionable. "Retrieval took 40ms and returned the right chunk
at rank 1, the model was called with 4.2k tokens of context, and it answered
with a figure that was not in the context" is actionable, and you only get that
from a trace.

Degrades to a no-op when OTEL is not installed or no endpoint is configured, so
nothing here can break a run.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ..config import get_settings

_tracer: Any = None
_initialised = False


def init_tracing() -> Any:
    """Configure the global tracer once. Returns None if tracing is off."""
    global _tracer, _initialised
    if _initialised:
        return _tracer
    _initialised = True

    s = get_settings()
    if not s.otlp_endpoint:
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return None

    provider = TracerProvider(
        resource=Resource.create({"service.name": s.service_name})
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{s.otlp_endpoint}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(s.service_name)
    return _tracer


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[dict[str, Any]]:
    """Trace a block. Always yields a dict the caller can add attributes to,
    whether or not a real tracer exists."""
    tracer = init_tracing()
    bag: dict[str, Any] = dict(attributes)
    started = time.perf_counter()

    if tracer is None:
        try:
            yield bag
        finally:
            bag["duration_ms"] = int((time.perf_counter() - started) * 1000)
        return

    with tracer.start_as_current_span(name) as otel_span:
        try:
            yield bag
        finally:
            bag["duration_ms"] = int((time.perf_counter() - started) * 1000)
            for key, value in bag.items():
                if isinstance(value, (str, int, float, bool)):
                    otel_span.set_attribute(key, value)


def traced(name: str | None = None) -> Callable:
    """Decorator form, for functions where the whole call is the unit of work."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(name or fn.__qualname__):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


# ------------------------------------------------------------ token ledger
@dataclass(slots=True)
class TokenLedger:
    """Per-operation token and cost accounting.

    Separate from tracing because the question "what did last month cost, split
    by operation" needs to be answerable without a trace backend. Finding that
    the judge costs more than the answers, or that one retry path is a third of
    the bill, is the kind of discovery this makes routine.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        operation: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
        latency_ms: int = 0,
    ) -> float:
        s = get_settings()
        cost = s.cost_usd(prompt_tokens, completion_tokens)
        self.entries.append(
            {
                "operation": operation,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": cost,
                "latency_ms": latency_ms,
            }
        )
        return cost

    @property
    def total_cost(self) -> float:
        return sum(e["cost_usd"] for e in self.entries)

    @property
    def total_tokens(self) -> int:
        return sum(e["total_tokens"] for e in self.entries)

    def by_operation(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for entry in self.entries:
            bucket = out.setdefault(
                entry["operation"],
                {"calls": 0, "tokens": 0, "cost_usd": 0.0, "latency_ms": 0},
            )
            bucket["calls"] += 1
            bucket["tokens"] += entry["total_tokens"]
            bucket["cost_usd"] += entry["cost_usd"]
            bucket["latency_ms"] += entry["latency_ms"]
        for bucket in out.values():
            bucket["cost_usd"] = round(bucket["cost_usd"], 6)
            bucket["avg_latency_ms"] = round(bucket["latency_ms"] / bucket["calls"], 1)
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "calls": len(self.entries),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "by_operation": self.by_operation(),
        }
