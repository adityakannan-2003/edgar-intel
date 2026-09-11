"""Agent tools: typed, validated, and deliberately few.

Tool design is where most agent projects go wrong, so the choices here are
explicit:

  * Six tools, not twenty. Every additional tool enlarges the space the model
    has to choose from and measurably degrades selection accuracy. A tool that
    is rarely the right call is a tool that is often the wrong one.
  * Every tool takes a Pydantic model and returns a Pydantic model. Arguments
    are validated before execution, so a hallucinated field is a clean
    validation error the agent can recover from rather than a traceback.
  * Tools return structured data plus a short `summary` string. The model reads
    the summary; the caller keeps the structure for citations and audit.
  * `escalate_to_human` is a first-class tool. Giving the model an explicit,
    cheap way to stop is what keeps it from inventing an answer when the
    evidence is not there.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .. import db
from ..ingest.xbrl import CORE_TAGS, format_value
from ..retrieval.search import search


# ----------------------------------------------------------------- schemas
class SearchFilingsArgs(BaseModel):
    query: str = Field(description="Natural-language search over filing text.")
    ticker: str | None = Field(default=None, description="Restrict to one company, e.g. AAPL.")
    fiscal_year: int | None = Field(default=None, description="Restrict to one fiscal year.")
    item: str | None = Field(default=None, description="Restrict to a 10-K Item, e.g. 1A or 7.")
    top_n: int = Field(default=5, ge=1, le=20)

    @field_validator("ticker")
    @classmethod
    def upper(cls, v: str | None) -> str | None:
        return v.upper() if v else v


class GetFactArgs(BaseModel):
    ticker: str
    tag: str = Field(description="XBRL tag, e.g. Revenues or NetIncomeLoss.")
    fiscal_year: int
    fiscal_period: str = Field(default="FY")

    @field_validator("ticker")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.upper()


class CompareFactArgs(BaseModel):
    ticker: str
    tag: str
    year_a: int
    year_b: int

    @field_validator("ticker")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.upper()


class RatioArgs(BaseModel):
    ticker: str
    numerator_tag: str
    denominator_tag: str
    fiscal_year: int

    @field_validator("ticker")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.upper()


class ListCoverageArgs(BaseModel):
    ticker: str | None = None


class EscalateArgs(BaseModel):
    reason: str = Field(description="Why this needs a human, in one sentence.")
    partial_findings: str = Field(default="", description="Anything useful found so far.")


class ToolResult(BaseModel):
    ok: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------- tools
def search_filings(args: SearchFilingsArgs) -> ToolResult:
    result = search(
        args.query,
        ticker=args.ticker,
        fiscal_year=args.fiscal_year,
        item=args.item,
        top_n=args.top_n,
    )
    if not result.hits:
        return ToolResult(
            ok=False,
            summary="No passages matched. Try a broader query or drop the filters.",
            data={"filters": args.model_dump()},
        )
    passages = [
        {
            "id": h.chunk_id,
            "ticker": h.ticker,
            "fiscal_year": h.fiscal_year,
            "item": h.item,
            # Truncated deliberately: a tool returning 8k characters per hit
            # blows the context budget in two steps, leaving the agent no room
            # to reason.
            "text": h.body[:1200],
        }
        for h in result.hits
    ]
    summary = "\n\n".join(
        f"[{p['id']}] {p['ticker']} FY{p['fiscal_year']} Item {p['item']}\n{p['text'][:500]}"
        for p in passages
    )
    return ToolResult(
        ok=True,
        summary=summary,
        data={"passages": passages, "latency_ms": result.latency_ms},
        citations=[p["id"] for p in passages],
    )


def get_fact(args: GetFactArgs) -> ToolResult:
    row = db.query_one(
        """
        SELECT x.value, x.unit, x.tag, x.fiscal_year, x.accession, co.name
          FROM xbrl_facts x
          JOIN companies co ON co.cik = x.cik
         WHERE co.ticker = %s AND x.tag = %s
           AND x.fiscal_year = %s AND x.fiscal_period = %s
         LIMIT 1
        """,
        (args.ticker, args.tag, args.fiscal_year, args.fiscal_period),
    )
    if not row:
        available = db.query(
            """
            SELECT DISTINCT x.tag
              FROM xbrl_facts x JOIN companies co ON co.cik = x.cik
             WHERE co.ticker = %s AND x.fiscal_year = %s
             LIMIT 15
            """,
            (args.ticker, args.fiscal_year),
        )
        return ToolResult(
            ok=False,
            summary=(
                f"No {args.tag} reported for {args.ticker} FY{args.fiscal_year}. "
                f"Available tags: {', '.join(r['tag'] for r in available) or 'none'}"
            ),
            data={"available_tags": [r["tag"] for r in available]},
        )
    value = float(row["value"])
    return ToolResult(
        ok=True,
        summary=(
            f"{row['name']} {CORE_TAGS.get(row['tag'], row['tag'])} FY{row['fiscal_year']}: "
            f"{format_value(value, row['unit'])} (source: XBRL {row['accession']})"
        ),
        data={
            "value": value,
            "unit": row["unit"],
            "tag": row["tag"],
            "fiscal_year": row["fiscal_year"],
            "accession": row["accession"],
        },
        citations=[f"xbrl:{row['accession']}"],
    )


def compare_fact(args: CompareFactArgs) -> ToolResult:
    rows = db.query(
        """
        SELECT x.fiscal_year, x.value, x.unit
          FROM xbrl_facts x JOIN companies co ON co.cik = x.cik
         WHERE co.ticker = %s AND x.tag = %s
           AND x.fiscal_year = ANY(%s) AND x.fiscal_period = 'FY'
        """,
        (args.ticker, args.tag, [args.year_a, args.year_b]),
    )
    by_year = {int(r["fiscal_year"]): float(r["value"]) for r in rows}
    if args.year_a not in by_year or args.year_b not in by_year:
        missing = [y for y in (args.year_a, args.year_b) if y not in by_year]
        return ToolResult(
            ok=False,
            summary=f"Missing {args.tag} for {args.ticker} in {missing}.",
            data={"have": sorted(by_year)},
        )
    a, b = by_year[args.year_a], by_year[args.year_b]
    if a == 0:
        return ToolResult(ok=False, summary="Base year value is zero; percentage change undefined.")
    pct = (b - a) / abs(a) * 100
    unit = rows[0]["unit"]
    direction = "increased" if pct >= 0 else "decreased"
    return ToolResult(
        ok=True,
        summary=(
            f"{args.ticker} {args.tag}: FY{args.year_a} {format_value(a, unit)} -> "
            f"FY{args.year_b} {format_value(b, unit)}, {direction} {abs(pct):.1f}%"
        ),
        data={"year_a": a, "year_b": b, "pct_change": round(pct, 4), "unit": unit},
    )


def compute_ratio(args: RatioArgs) -> ToolResult:
    rows = db.query(
        """
        SELECT x.tag, x.value, x.unit
          FROM xbrl_facts x JOIN companies co ON co.cik = x.cik
         WHERE co.ticker = %s AND x.tag = ANY(%s)
           AND x.fiscal_year = %s AND x.fiscal_period = 'FY'
        """,
        (args.ticker, [args.numerator_tag, args.denominator_tag], args.fiscal_year),
    )
    values = {r["tag"]: float(r["value"]) for r in rows}
    if args.numerator_tag not in values or args.denominator_tag not in values:
        return ToolResult(
            ok=False,
            summary=f"Need both {args.numerator_tag} and {args.denominator_tag}; have {list(values)}.",
        )
    denom = values[args.denominator_tag]
    if denom == 0:
        return ToolResult(ok=False, summary="Denominator is zero.")
    ratio = values[args.numerator_tag] / denom
    return ToolResult(
        ok=True,
        summary=(
            f"{args.ticker} FY{args.fiscal_year}: {args.numerator_tag} / "
            f"{args.denominator_tag} = {ratio:.4f} ({ratio * 100:.2f}%)"
        ),
        data={"ratio": round(ratio, 6), "pct": round(ratio * 100, 4), **values},
    )


def list_coverage(args: ListCoverageArgs) -> ToolResult:
    """What the system actually holds.

    Present so the agent can discover the boundary of its own knowledge instead
    of guessing about companies or years that were never ingested. Most
    hallucinations at the agent layer are really coverage questions the agent
    had no way to ask.
    """
    if args.ticker:
        rows = db.query(
            """
            SELECT co.ticker, co.name, f.fiscal_year, f.form
              FROM filings f JOIN companies co ON co.cik = f.cik
             WHERE co.ticker = %s
             ORDER BY f.fiscal_year DESC
            """,
            (args.ticker.upper(),),
        )
    else:
        rows = db.query(
            """
            SELECT co.ticker, co.name,
                   MIN(f.fiscal_year) AS first_year,
                   MAX(f.fiscal_year) AS last_year,
                   COUNT(*) AS filings
              FROM filings f JOIN companies co ON co.cik = f.cik
             GROUP BY co.ticker, co.name
             ORDER BY co.ticker
            """
        )
    if not rows:
        return ToolResult(ok=False, summary="No filings ingested yet.")
    lines = [", ".join(f"{k}={v}" for k, v in r.items()) for r in rows]
    return ToolResult(ok=True, summary="\n".join(lines), data={"rows": rows})


def escalate_to_human(args: EscalateArgs) -> ToolResult:
    return ToolResult(
        ok=True,
        summary=f"ESCALATED: {args.reason}",
        data={"reason": args.reason, "partial_findings": args.partial_findings},
    )


# ------------------------------------------------------------------ registry
TOOLS: dict[str, dict[str, Any]] = {
    "search_filings": {
        "fn": search_filings,
        "args": SearchFilingsArgs,
        "description": (
            "Search filing narrative text (Risk Factors, MD&A, Business) "
            "for passages."
        ),
    },
    "get_fact": {
        "fn": get_fact,
        "args": GetFactArgs,
        "description": (
            "Look up one reported XBRL financial figure. Authoritative; "
            "prefer this over text search for any number."
        ),
    },
    "compare_fact": {
        "fn": compare_fact,
        "args": CompareFactArgs,
        "description": (
            "Compare one XBRL figure across two fiscal years and get the "
            "percentage change."
        ),
    },
    "compute_ratio": {
        "fn": compute_ratio,
        "args": RatioArgs,
        "description": "Divide one XBRL figure by another for the same company and year.",
    },
    "list_coverage": {
        "fn": list_coverage,
        "args": ListCoverageArgs,
        "description": (
            "List which companies and fiscal years are available. Use when "
            "unsure of coverage."
        ),
    },
    "escalate_to_human": {
        "fn": escalate_to_human,
        "args": EscalateArgs,
        "description": (
            "Stop and hand off to a human. Use when evidence is missing or "
            "the question is out of scope."
        ),
    },
}


def tool_catalogue() -> str:
    """Tool descriptions plus JSON schemas, rendered for the system prompt."""
    lines = []
    for name, spec in TOOLS.items():
        schema = spec["args"].model_json_schema()
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        fields = ", ".join(
            f"{k}{'' if k in required else '?'}: {v.get('type', 'any')}"
            for k, v in props.items()
        )
        lines.append(f"- {name}({fields})\n    {spec['description']}")
    return "\n".join(lines)


def call_tool(name: str, raw_args: dict[str, Any]) -> ToolResult:
    """Validate then execute. Validation failures return a usable error."""
    spec = TOOLS.get(name)
    if spec is None:
        return ToolResult(
            ok=False,
            summary=f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}",
        )
    try:
        args = spec["args"](**raw_args)
    except Exception as exc:
        return ToolResult(ok=False, summary=f"Invalid arguments for {name}: {exc}")
    try:
        return spec["fn"](args)
    except Exception as exc:
        return ToolResult(ok=False, summary=f"{name} failed: {exc}")
