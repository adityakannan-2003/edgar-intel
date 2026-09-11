"""FastAPI service.

Small on purpose. The interesting engineering is upstream; what this layer has
to get right is the operational surface: health that distinguishes liveness
from readiness, request ids that tie an HTTP response to an agent trace, and
per-request timing and cost returned in the response so the client can see what
it spent.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .. import db
from ..agent.loop import outcome_stats, run_agent
from ..config import get_settings
from ..db import close_pool
from ..evals.report import latest_run
from ..obs.tracing import init_tracing, span
from ..retrieval.search import search


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown.

    `@app.on_event` is deprecated in current FastAPI; the lifespan context
    manager is the supported form and, unlike the old hooks, guarantees the
    teardown half runs even when startup raises.
    """
    init_tracing()
    yield
    close_pool()


app = FastAPI(
    title="edgar-intel",
    version="0.1.0",
    description="Retrieval, evaluation and agent API over SEC filings.",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = str(int((time.perf_counter() - started) * 1000))
    return response


# ------------------------------------------------------------------ models
class SearchRequest(BaseModel):
    query: str
    ticker: str | None = None
    fiscal_year: int | None = None
    item: str | None = None
    mode: str = Field(default="hybrid", pattern="^(dense|lexical|hybrid)$")
    strategy: str | None = None
    top_n: int = Field(default=8, ge=1, le=50)
    rerank: bool = True


class AskRequest(BaseModel):
    question: str
    max_steps: int | None = Field(default=None, ge=1, le=20)
    confidence_floor: float | None = Field(default=None, ge=0.0, le=1.0)


# ----------------------------------------------------------------- routes
@app.get("/health")
def health() -> dict[str, str]:
    """Liveness. Answers 'is the process up', nothing more."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    """Readiness. Checks the dependencies a request actually needs.

    Kept distinct from /health deliberately: an orchestrator that restarts the
    process because the database blinked has made an outage worse, not better.
    """
    checks = {"database": db.healthcheck()}
    try:
        row = db.query_one("SELECT COUNT(*) AS n FROM chunks WHERE embedding IS NOT NULL")
        checks["index"] = bool(row and row["n"] > 0)
        checks["indexed_chunks"] = int(row["n"]) if row else 0
    except Exception:
        checks["index"] = False

    ok = bool(checks.get("database")) and bool(checks.get("index"))
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"ready": ok, "checks": checks},
    )


@app.post("/search")
def search_endpoint(req: SearchRequest) -> dict[str, Any]:
    with span("api.search", mode=req.mode, ticker=req.ticker or "") as bag:
        result = search(
            req.query,
            strategy=req.strategy,
            mode=req.mode,
            use_rerank=req.rerank,
            top_n=req.top_n,
            ticker=req.ticker,
            fiscal_year=req.fiscal_year,
            item=req.item,
        )
        bag["hits"] = len(result.hits)
    return {
        "query": req.query,
        "latency_ms": result.latency_ms,
        "stage_latency_ms": result.stage_latency_ms,
        "hits": [
            {**h.as_dict(), "text": h.body[:1000]}
            for h in result.hits
        ],
    }


@app.post("/ask")
def ask_endpoint(req: AskRequest) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    with span("api.ask") as bag:
        run = run_agent(
            req.question,
            max_steps=req.max_steps,
            confidence_floor=req.confidence_floor,
        )
        bag["outcome"] = run.outcome
        bag["steps"] = len(run.steps)
    return {
        "trace_id": run.trace_id,
        "outcome": run.outcome,
        "answer": run.answer,
        "citations": run.citations,
        "confidence": run.confidence,
        "steps": len(run.steps),
        "latency_ms": run.total_latency_ms,
        "cost_usd": run.total_cost_usd,
    }


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM agent_traces WHERE trace_id = %s", (trace_id,))
    if not row:
        raise HTTPException(status_code=404, detail="trace not found")
    return dict(row)


@app.get("/stats/agent")
def agent_stats() -> dict[str, Any]:
    return outcome_stats()


@app.get("/stats/eval")
def eval_stats() -> dict[str, Any]:
    """Latest evaluation summary, served next to the thing it evaluates.

    Exposing quality metrics on the same service that answers questions means
    "how good is this right now" is a request away rather than a dashboard
    someone has to remember to open.
    """
    run = latest_run()
    if not run:
        return {"status": "no evaluation runs recorded"}
    summary = run.get("summary")
    if isinstance(summary, str):
        import json

        summary = json.loads(summary)
    return {"run_key": run["run_key"], "label": run["label"], "summary": summary}


@app.get("/config")
def config() -> dict[str, Any]:
    """Non-secret runtime config. Answers 'what is this instance actually running'."""
    s = get_settings()
    return {
        "strategy": s.default_strategy,
        "retrieve_k": s.retrieve_k,
        "rerank_top_n": s.rerank_top_n,
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model,
        "embed_provider": s.embed_provider,
        "embed_model": s.embed_model,
        "agent_max_steps": s.agent_max_steps,
        "agent_confidence_floor": s.agent_confidence_floor,
    }
