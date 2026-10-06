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
    tag: str = Field(
        description=(
            "XBRL tag, e.g. Revenues or NetIncomeLoss. 'revenue' is accepted and "
            "resolves to the revenue tag the company reports."
        )
    )
    fiscal_year: int
    fiscal_period: str = Field(default="FY")

    @field_validator("ticker")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.upper()


class CompareFactArgs(BaseModel):
    ticker: str
    tag: str = Field(
        description=(
            "XBRL tag, e.g. Revenues or NetIncomeLoss. 'revenue' is accepted and "
            "resolves to the revenue tag the company reports in both years."
        )
    )
    year_a: int
    year_b: int

    @field_validator("ticker")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.upper()


class RatioArgs(BaseModel):
    ticker: str
    numerator_tag: str = Field(description="XBRL tag. 'revenue' is accepted.")
    denominator_tag: str = Field(description="XBRL tag. 'revenue' is accepted.")
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


# The model passes the user's word for a figure as the tag. No XBRL fact is
# tagged "revenue", so before this map `get_fact` matched nothing and the agent
# escalated questions whose answer was in the database. `compare_fact` and
# `compute_ratio` had the same gap and resolve through the same map.
#
# Deterministic by construction: a word maps to a fixed, ordered list of tags,
# and the first one the company reported wins -- for that year in `get_fact` and
# `compute_ratio`, for both years in `compare_fact`. The order is
# `PREFERRED_TAG_FAMILIES["revenue"]` in evals/goldenset.py -- the tag the golden
# set grades against -- and a test holds the two together.
TAG_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        # CAT, NVDA, PG and UNH report revenue only under this one.
        "Revenues",
    ),
}


def resolve_tag(tag: str) -> tuple[str, ...]:
    """The XBRL tags to try for `tag`, in order.

    A real tag passes through untouched: `Revenues` means `Revenues`, even for a
    company that also reports the contract tag. Only a word that is not a tag
    is looked up, ignoring case and surrounding whitespace.
    """
    if tag in CORE_TAGS:
        return (tag,)
    return TAG_ALIASES.get(tag.strip().lower(), (tag,))


def _tried(tag: str, tags: tuple[str, ...]) -> str:
    """` (tried A, B)` when `tag` was an alias, so a miss names what was looked up."""
    return f" (tried {', '.join(tags)})" if tags != (tag,) else ""


def get_fact(args: GetFactArgs) -> ToolResult:
    tags = resolve_tag(args.tag)
    row = None
    for tag in tags:
        row = db.query_one(
            """
            SELECT x.value, x.unit, x.tag, x.fiscal_year, x.accession, co.name
              FROM xbrl_facts x
              JOIN companies co ON co.cik = x.cik
             WHERE co.ticker = %s AND x.tag = %s
               AND x.fiscal_year = %s AND x.fiscal_period = %s
             LIMIT 1
            """,
            (args.ticker, tag, args.fiscal_year, args.fiscal_period),
        )
        if row:
            break
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
                f"No {args.tag} reported for {args.ticker} "
                f"FY{args.fiscal_year}{_tried(args.tag, tags)}. "
                f"Available tags: {', '.join(r['tag'] for r in available) or 'none'}"
            ),
            data={"available_tags": [r["tag"] for r in available]},
        )
    value = float(row["value"])
    # The model cites what it reads, so the summary prints the id exactly as
    # registered. It used to print "XBRL <accession>" and register
    # "xbrl:<accession>": the model cited the bare accession, the citation check
    # rejected it, and 7 of 20 pilot questions escalated with the right figure
    # in hand (docs/METRICS.md §6).
    citation = f"xbrl:{row['accession']}"
    return ToolResult(
        ok=True,
        summary=(
            f"{row['name']} {CORE_TAGS.get(row['tag'], row['tag'])} FY{row['fiscal_year']}: "
            f"{format_value(value, row['unit'])} (source: {citation})"
        ),
        data={
            "value": value,
            "unit": row["unit"],
            "tag": row["tag"],
            "fiscal_year": row["fiscal_year"],
            "accession": row["accession"],
        },
        citations=[citation],
    )


def compare_fact(args: CompareFactArgs) -> ToolResult:
    # One tag for both years, never one per year: a change between two different
    # concepts is not a change, and the golden set only builds year-over-year
    # cases within a single tag. A company that switched tags still gets an
    # answer from the next tag in order, if that one covers both years.
    tags = resolve_tag(args.tag)
    rows: list[dict[str, Any]] = []
    by_year: dict[int, float] = {}
    have: set[int] = set()
    for tag in tags:
        rows = db.query(
            """
            SELECT x.fiscal_year, x.value, x.unit, x.prior_year_value
              FROM xbrl_facts x JOIN companies co ON co.cik = x.cik
             WHERE co.ticker = %s AND x.tag = %s
               AND x.fiscal_year = ANY(%s) AND x.fiscal_period = 'FY'
            """,
            (args.ticker, tag, [args.year_a, args.year_b]),
        )
        by_year = {int(r["fiscal_year"]): float(r["value"]) for r in rows}
        have.update(by_year)
        if args.year_a in by_year and args.year_b in by_year:
            break
    if args.year_a not in by_year or args.year_b not in by_year:
        missing = [y for y in (args.year_a, args.year_b) if y not in have]
        tried = _tried(args.tag, tags)
        return ToolResult(
            ok=False,
            summary=(
                f"Missing {args.tag} for {args.ticker} in {missing}{tried}."
                if missing
                else f"{args.tag} for {args.ticker} is reported under different tags in "
                f"FY{args.year_a} and FY{args.year_b}{tried}, so there is no "
                "like-for-like change."
            ),
            data={"have": sorted(have)},
        )
    # Adjacent years go on the later year's basis -- its filing's own figure
    # for the year before -- so a stock split or restatement between the two
    # filings does not read as a change (D12).
    earlier, later = sorted((args.year_a, args.year_b))
    printed = {int(r["fiscal_year"]): r["prior_year_value"] for r in rows}
    if later - earlier == 1 and printed.get(later) is not None:
        by_year[earlier] = float(printed[later])
    a, b = by_year[args.year_a], by_year[args.year_b]
    if a == 0:
        return ToolResult(ok=False, summary="Base year value is zero; percentage change undefined.")
    pct = (b - a) / abs(a) * 100
    unit = rows[0]["unit"]
    direction = "increased" if pct >= 0 else "decreased"
    return ToolResult(
        ok=True,
        summary=(
            f"{args.ticker} {tag}: FY{args.year_a} {format_value(a, unit)} -> "
            f"FY{args.year_b} {format_value(b, unit)}, {direction} {abs(pct):.1f}%"
        ),
        data={"year_a": a, "year_b": b, "pct_change": round(pct, 4), "unit": unit, "tag": tag},
    )


def compute_ratio(args: RatioArgs) -> ToolResult:
    numerators = resolve_tag(args.numerator_tag)
    denominators = resolve_tag(args.denominator_tag)
    rows = db.query(
        """
        SELECT x.tag, x.value, x.unit
          FROM xbrl_facts x JOIN companies co ON co.cik = x.cik
         WHERE co.ticker = %s AND x.tag = ANY(%s)
           AND x.fiscal_year = %s AND x.fiscal_period = 'FY'
        """,
        (args.ticker, [*numerators, *denominators], args.fiscal_year),
    )
    values = {r["tag"]: float(r["value"]) for r in rows}
    # Each side takes the first of its tags that was reported, in `resolve_tag`
    # order -- not whichever row the database returned first.
    num = next((t for t in numerators if t in values), None)
    den = next((t for t in denominators if t in values), None)
    if num is None or den is None:
        return ToolResult(
            ok=False,
            summary=(
                f"Need both {args.numerator_tag}{_tried(args.numerator_tag, numerators)} and "
                f"{args.denominator_tag}{_tried(args.denominator_tag, denominators)}; "
                f"have {list(values)}."
            ),
        )
    denom = values[den]
    if denom == 0:
        return ToolResult(ok=False, summary="Denominator is zero.")
    ratio = values[num] / denom
    return ToolResult(
        ok=True,
        summary=(
            f"{args.ticker} FY{args.fiscal_year}: {num} / {den} = {ratio:.4f} "
            f"({ratio * 100:.2f}%)"
        ),
        data={
            "ratio": round(ratio, 6),
            "pct": round(ratio * 100, 4),
            num: values[num],
            den: denom,
        },
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
