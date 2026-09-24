"""Command line interface.

One entry point for every operation in the system, because a project you
operate through a README full of Python snippets is a project nobody else can
run -- including you, six weeks later.
"""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import db
from .config import get_settings

app = typer.Typer(add_completion=False, help="edgar-intel: filings retrieval, evals and agent.")
console = Console()

ingest_app = typer.Typer(help="Fetch filings and XBRL facts.")
index_app = typer.Typer(help="Chunk and embed the corpus.")
eval_app = typer.Typer(help="Golden set, evaluation runs, and the regression gate.")
agent_app = typer.Typer(help="Ask the bounded agent.")
ft_app = typer.Typer(help="Fine-tuning dataset, training and benchmark.")
bench_app = typer.Typer(help="Serving benchmarks.")
mcp_app = typer.Typer(help="MCP server.")
fixture_app = typer.Typer(help="Frozen test corpus.")

app.add_typer(ingest_app, name="ingest")
app.add_typer(index_app, name="index")
app.add_typer(eval_app, name="eval")
app.add_typer(agent_app, name="agent")
app.add_typer(ft_app, name="finetune")
app.add_typer(bench_app, name="bench")
app.add_typer(mcp_app, name="mcp")
app.add_typer(fixture_app, name="fixture")


def _table(title: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        console.print(f"[yellow]{title}: nothing to show[/yellow]")
        return
    table = Table(title=title, header_style="bold")
    for column in rows[0]:
        table.add_column(str(column))
    for row in rows:
        table.add_row(*[_fmt(v) for v in row.values()])
    console.print(table)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


# --------------------------------------------------------------------- init
@app.command("init")
def init_db(schema: str = "sql") -> None:
    """Apply the database schema (every sql/*.sql file, in order)."""
    for path in db.apply_schema(schema):
        console.print(f"[green]applied[/green] {path}")


@app.command("status")
def status() -> None:
    """What is in the database right now."""
    if not db.healthcheck():
        console.print("[red]database unreachable[/red]")
        raise typer.Exit(1)
    counts = db.query(
        """
        SELECT 'companies' AS entity, COUNT(*) AS n FROM companies
        UNION ALL SELECT 'filings', COUNT(*) FROM filings
        UNION ALL SELECT 'sections', COUNT(*) FROM sections
        UNION ALL SELECT 'chunks', COUNT(*) FROM chunks
        UNION ALL SELECT 'xbrl_facts', COUNT(*) FROM xbrl_facts
        UNION ALL SELECT 'eval_runs', COUNT(*) FROM eval_runs
        UNION ALL SELECT 'agent_traces', COUNT(*) FROM agent_traces
        """
    )
    _table("corpus", counts)

    from .retrieval.index import index_report

    _table("index by strategy", index_report())


# ------------------------------------------------------------------- ingest
@ingest_app.command("run")
def ingest_run(
    tickers: str = typer.Option("", help="Comma-separated tickers; default is the configured universe."),
    forms: str = typer.Option("10-K"),
    years: int = typer.Option(3),
) -> None:
    from .ingest.pipeline import ingest_universe

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()] or None
    with console.status("fetching from EDGAR..."):
        stats = ingest_universe(
            ticker_list,
            [f.strip() for f in forms.split(",")],
            years,
            progress=lambda msg: console.log(msg),
        )
    console.print_json(json.dumps(stats.as_dict(), indent=2))


@ingest_app.command("doctor")
def ingest_doctor(
    tickers: str = typer.Option("", help="Comma-separated tickers; default is the universe."),
    form: str = typer.Option("10-K"),
    save_html: bool = typer.Option(False, help="Keep the raw HTML for offline debugging."),
    html_dir: str = typer.Option("data/raw_filings"),
    report: str = typer.Option("reports/parser_doctor.json"),
) -> None:
    """Check the parser against real filings without touching the database.

    Run this BEFORE `ingest run`. The parser works on the fixture corpus; real
    filings are inconsistent between filers and years, and when parsing fails it
    fails quietly — sections come back fused or empty, retrieval degrades, and
    it looks like a model problem. This turns that into a report that names the
    filing and the fix.
    """
    from .ingest import doctor

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()] or None
    with console.status("fetching filings from EDGAR..."):
        payload = doctor.run(
            ticker_list,
            form,
            save_html_dir=html_dir if save_html else None,
            progress=lambda msg: console.log(msg),
        )

    for entry in payload["diagnoses"]:
        diag = doctor.FilingDiagnosis(
            ticker=entry["ticker"],
            accession=entry["accession"],
            fiscal_year=entry["fiscal_year"],
            html_bytes=entry["html_bytes"],
            text_chars=entry["text_chars"],
            items_found=entry["items_found"],
            sections=entry["sections"],
            checks=[doctor.Check(**c) for c in entry["checks"]],
            error=entry["error"],
        )
        colour = "green" if diag.healthy else "red"
        console.print(f"[{colour}]{diag.render()}[/{colour}]")
        console.print()

    path = doctor.save_report(payload, report)
    console.print(
        f"[bold]{payload['healthy']}/{payload['filings_checked']} filings parsed cleanly[/bold]"
    )
    console.print(payload["verdict"])
    console.print(f"[dim]full report: {path}[/dim]")
    if payload["broken"]:
        raise typer.Exit(1)


@ingest_app.command("verify-facts")
def ingest_verify_facts() -> None:
    """Check the XBRL ground truth against the filing index.

    Run this after every `ingest run`. The golden set's expected answers come
    straight from `xbrl_facts`, so a wrong fiscal year there is not a bug that
    shows up as an error — it shows up as the model being marked wrong, at
    whatever rate the corruption happens to reach, and every hour spent tuning
    retrieval afterwards is wasted.

    Two independent checks. First, no fiscal year may hold two different values
    for the same tag; that is impossible if the fiscal year was derived from the
    fact's own period and inevitable if it was read from the report's `fy`.
    Second, each fact's period end must match the period end that the filing
    index — a completely separate SEC endpoint — records for that fiscal year.
    Agreement between the two is what makes the derivation rule evidence rather
    than an assumption.
    """
    rows = db.query(
        """
        SELECT co.ticker, f.tag, f.fiscal_year, COUNT(DISTINCT f.value) AS values,
               array_agg(DISTINCT f.value::float8) AS vals
          FROM xbrl_facts f JOIN companies co ON co.cik = f.cik
         WHERE f.fiscal_period = 'FY'
         GROUP BY 1, 2, 3
        HAVING COUNT(DISTINCT f.value) > 1
         ORDER BY 1, 2, 3
        """
    )
    if rows:
        console.print(f"[red]{len(rows)} (ticker, tag, year) groups hold conflicting values[/red]")
        _table("conflicting facts", [dict(r) for r in rows][:20])
    else:
        console.print("[green]no fiscal year holds two values for the same tag[/green]")

    drift = db.query(
        """
        SELECT co.ticker, x.fiscal_year, x.tag, x.period_end AS fact_end,
               fl.period_end AS filing_end
          FROM xbrl_facts x
          JOIN companies co ON co.cik = x.cik
          JOIN filings fl ON fl.cik = x.cik AND fl.fiscal_year = x.fiscal_year
         WHERE x.fiscal_period = 'FY'
           AND fl.period_end IS NOT NULL
           AND ABS(x.period_end - fl.period_end) > 7
         ORDER BY 1, 2
         LIMIT 20
        """
    )
    if drift:
        console.print(
            f"[red]{len(drift)} facts are dated more than a week from the filing's "
            "own period end — the fiscal-year rule does not hold for this filer[/red]"
        )
        _table("period drift", [dict(r) for r in drift])
    else:
        console.print("[green]every fact's period end agrees with the filing index[/green]")

    if rows or drift:
        raise typer.Exit(1)


# -------------------------------------------------------------------- index
@index_app.command("build")
def index_build(
    strategy: str = typer.Option("all", help="fixed | recursive | section_aware | semantic | all"),
    target_tokens: int = typer.Option(512),
    overlap: int = typer.Option(64),
) -> None:
    from .chunking import STRATEGIES
    from .retrieval.index import index_strategy

    strategies = list(STRATEGIES) if strategy == "all" else [strategy]
    rows = []
    for st in strategies:
        with console.status(f"building {st}..."):
            stats = index_strategy(st, target_tokens, overlap, progress=lambda m: console.log(m))
        rows.append(stats.as_dict())
    _table("index build", rows)


@index_app.command("report")
def index_report_cmd() -> None:
    from .retrieval.index import index_report

    _table("index by strategy", index_report())


# --------------------------------------------------------------------- eval
@eval_app.command("build")
def eval_build(
    path: str = typer.Option("evalset/golden.json"),
    numeric: int = typer.Option(20, help="Numeric cases per company."),
    narrative: int = typer.Option(3, help="Narrative cases per company."),
    strategy: str = typer.Option(""),
) -> None:
    """Generate the golden set from XBRL facts plus narrative seeds."""
    import os

    from .evals.goldenset import build

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cases = build(path, numeric, narrative, strategy or None)
    kinds: dict[str, int] = {}
    linked = 0
    for case in cases:
        kinds[case.kind] = kinds.get(case.kind, 0) + 1
        if case.relevant_chunk_ids:
            linked += 1
    console.print(f"[green]{len(cases)} cases written to {path}[/green]")
    console.print(f"  by kind: {kinds}")
    console.print(f"  with linked evidence: {linked}/{len(cases)}")
    if linked < len(cases) * 0.5:
        console.print(
            "[yellow]Fewer than half the cases have linked evidence. Retrieval "
            "metrics will be computed on a small subset -- build the index first, "
            "or widen the evidence linker.[/yellow]"
        )


@eval_app.command("narrative-sources")
def eval_narrative_sources(
    per_company: int = typer.Option(5, help="Narrative topics per company to plan for."),
    missing_only: bool = typer.Option(
        True, help="Dump only cases that have no human reference yet."
    ),
    cases: str = typer.Option("", help="Comma-separated case ids; overrides --missing-only."),
    out: str = typer.Option("", help="Output path. Empty = a name derived from the selection."),
    overwrite: bool = typer.Option(False, help="Replace an existing dump."),
    max_section_chars: int = typer.Option(
        0, help="Truncate each section, marked explicitly. 0 = no truncation."
    ),
) -> None:
    """Dump the filing sections a narrative reference answer must be written from.

    Expanding the narrative set to 8 companies x 5 topics needs 16 new reference
    answers, and a reference answer is ground truth: written from memory it
    measures the writer's recollection instead of the system. Every existing
    reference was written against `reports/narrative_source_sections.txt`; this
    is that dump, reproducible, for the cases that do not have one yet.

    A case whose expected Item has no section is reported rather than skipped.
    Incorporation by reference is routine, and "no section" and "the company
    discloses nothing" must not look alike.
    """
    from .evals.narrative_sources import (
        default_dump_name,
        dump,
        missing_references,
        narrative_plan,
    )

    plan = narrative_plan(max_per_company=per_company)
    if cases:
        wanted = {c.strip() for c in cases.split(",") if c.strip()}
        selected = [p for p in plan if p.case_id in wanted]
        unknown = wanted - {p.case_id for p in selected}
        if unknown:
            console.print(f"[red]not in the plan: {', '.join(sorted(unknown))}[/red]")
            raise typer.Exit(1)
        label = "selected"
    elif missing_only:
        selected = missing_references(plan)
        label = "missing"
    else:
        selected = plan
        label = "all"

    if not selected:
        console.print("[green]every planned case already has a human reference[/green]")
        return

    def progress(sources) -> None:
        mark = (
            "[green]ok[/green]"
            if sources.expected_item_found
            else "[yellow]NO SECTION[/yellow]"
        )
        console.print(
            f"  {sources.plan.case_id:<34} item {sources.plan.item:<3} "
            f"{len(sources.sections)} sec  {sources.total_chars:>8,}c  {mark}"
        )

    try:
        payload = dump(
            selected,
            out_path=out or default_dump_name(len(selected), label),
            overwrite=overwrite,
            max_section_chars=max_section_chars,
            progress=progress,
        )
    except FileExistsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print()
    _table("narrative sources", payload["rows"])
    console.print(
        f"[bold]{payload['cases']} cases, {payload['total_chars']:,} chars -> "
        f"{payload['out_path']}[/bold]"
    )
    if payload["missing_sections"]:
        console.print(
            "[yellow]no section for the expected Item: "
            + ", ".join(payload["missing_sections"])
            + " -- the disclosure is elsewhere or incorporated by reference[/yellow]"
        )
    if payload["year_mismatches"]:
        console.print(
            "[yellow]case year and 10-K year disagree: "
            + ", ".join(payload["year_mismatches"])
            + " -- the dump names the year the text came from[/yellow]"
        )


@eval_app.command("run")
def eval_run(
    path: str = typer.Option("evalset/golden.json"),
    label: str = typer.Option(""),
    strategy: str = typer.Option(""),
    mode: str = typer.Option("hybrid"),
    rerank: bool = typer.Option(True),
    top_n: int = typer.Option(0, help="Passages placed in the answering context. 0 = configured default."),
    limit: int = typer.Option(0, help="Run only the first N cases."),
    git_sha: str = typer.Option(""),
) -> None:
    from .evals.goldenset import load
    from .evals.runner import RunAborted, run_suite

    cases = load(path)
    if limit:
        cases = cases[:limit]

    def progress(i: int, total: int, result) -> None:
        mark = "." if result.passed else ("~" if result.abstained else "F")
        console.print(mark, end="")
        if i % 50 == 0 or i == total:
            console.print(f" {i}/{total}")

    try:
        _, summary = run_suite(
            cases, label=label, strategy=strategy or None, mode=mode,
            use_rerank=rerank, top_n=top_n or None, sha=git_sha, progress=progress,
        )
    except RunAborted as exc:
        console.print()
        console.print(f"[red]run aborted[/red] {exc}")
        raise typer.Exit(2) from exc

    console.print()
    console.print_json(json.dumps(summary.as_dict(), indent=2))


@eval_app.command("gate")
def eval_gate(
    max_regression: float = typer.Option(0.03),
    baseline_label: str = typer.Option(""),
) -> None:
    """Fail the build if quality regressed beyond tolerance."""
    from .evals.report import gate

    result = gate(max_regression, baseline_label or None)
    console.print(result.render())
    if not result.passed:
        raise typer.Exit(1)


@eval_app.command("compare")
def eval_compare(
    path: str = typer.Option("evalset/golden.json"),
    strategies: str = typer.Option("fixed,recursive,section_aware"),
    limit: int = typer.Option(0),
) -> None:
    """Run the same cases under each chunking strategy and print the table."""
    from .evals.goldenset import load
    from .evals.report import compare_runs
    from .evals.runner import compare_strategies

    cases = load(path)
    if limit:
        cases = cases[:limit]
    summaries = compare_strategies(cases, [s.strip() for s in strategies.split(",")])
    _table("strategy comparison", compare_runs([s.run_key for s in summaries]))


@eval_app.command("context-probe")
def eval_context_probe(
    path: str = typer.Option("evalset/one_case.json", help="Case file to probe."),
    case_id: str = typer.Option("", help="Probe only this case id."),
    top_n: str = typer.Option("8,12,16,20", help="Comma-separated top_n grid."),
    rerank: str = typer.Option("on,off", help="Which rerank settings to test."),
    repeats: int = typer.Option(1, help="Repeats per arm; >1 tests reliability, not just pass/fail."),
    item_boost: float = typer.Option(-1.0, help="Override item_boost_weight. -1 = use configured value."),
    context_max_chars: int = typer.Option(12000, help="build_context character cap."),
    context_packing: str = typer.Option(
        "greedy-stop", help="greedy-stop (shipped) | skip-oversized"
    ),
    out: str = typer.Option("", help="Report path. Empty = a name derived from the arm."),
    overwrite: bool = typer.Option(False, help="Replace an existing report."),
) -> None:
    """Probe one case across a top_n x rerank grid and say where the evidence is lost.

    Built for the `nar-AAPL-legal` failure: all five relevant chunks sat inside
    the hybrid top-50 at ranks 9, 11, 13, 21 and 29, and `top_n=8` discarded
    every one before the model saw anything. The run reported a narrative
    quality failure. It was a context-selection failure.

    Changes nothing but `top_n` and the rerank flag between arms, records the
    config and git SHA each arm ran under, and writes to a report file rather
    than to `eval_runs` -- experiments do not belong in the baseline history.
    """
    from .evals.context_probe import probe
    from .evals.goldenset import load

    cases = load(path)
    if case_id:
        cases = [c for c in cases if c.case_id == case_id]
    if not cases:
        console.print(f"[red]no cases in {path}[/red]")
        raise typer.Exit(1)

    grid = tuple(int(x) for x in top_n.split(",") if x.strip())
    settings_map = {"on": True, "off": False}
    rerank_settings = tuple(
        settings_map[x.strip().lower()] for x in rerank.split(",") if x.strip()
    )

    def progress(arm) -> None:
        mark = "[green]PASS[/green]" if arm.passed else "[red]fail[/red]"
        console.print(
            f"  top_n={arm.top_n:<3} rerank={str(arm.use_rerank):<5} {mark}  "
            f"pre-trunc {arm.relevant_pre_truncation_ranks}  "
            f"after top_n {arm.relevant_in_context}/{arm.n_relevant}  "
            f"in prompt {arm.relevant_in_prompt}/{arm.n_relevant}  "
            f"hits {arm.n_included}/{len(arm.selected_ids)}  "
            f"{arm.total_ms} ms" + (f"  [red]{arm.error[:70]}[/red]" if arm.error else "")
        )

    try:
        with console.status("probing..."):
            payload = probe(
                cases,
                top_n_grid=grid,
                rerank_settings=rerank_settings,
                repeats=repeats,
                item_boost=None if item_boost < 0 else item_boost,
                context_max_chars=context_max_chars,
                context_packing=context_packing,
                out_path=out or None,
                overwrite=overwrite,
                progress=progress,
            )
    except FileExistsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print()
    _table("context probe", payload["rows"])
    console.print(f"[bold]{payload['recommendation']}[/bold]")
    console.print(f"[dim]full report: {payload['out_path']}[/dim]")


@eval_app.command("autopsy")
def eval_autopsy(
    path: str = typer.Option("evalset/one_case.json"),
    case_id: str = typer.Option("", help="Which case; defaults to the first in the file."),
    top_n: int = typer.Option(16),
    rerank: bool = typer.Option(False),
    item_boost: float = typer.Option(2.0),
    context_max_chars: int = typer.Option(24000),
    context_packing: str = typer.Option("greedy-stop"),
    facts: str = typer.Option("", help="label=regex; label=regex. Empty = derive from the reference."),
    replay: int = typer.Option(0, help="Regenerate N times from the saved prompt, retrieval bypassed."),
    judge_replay: int = typer.Option(0, help="Re-judge the saved answer N times."),
    judge_model: str = typer.Option("", help="Override the judge for the replay only."),
    show_context: bool = typer.Option(False, help="Print the exact context sent to the model."),
    out: str = typer.Option("reports/autopsy.json"),
) -> None:
    """Dissect one case end to end: prompt in, answer out, verdict, and why.

    Use when the context probe says the evidence reached the model and the case
    fails anyway. Everything downstream of retrieval is separable only if you
    can read the literal bytes at each boundary, so this prints them: the hits
    in prompt order, the exact context, the exact answer, the exact judge
    rationale, and a literal search of the prompt for every fact the reference
    requires -- searched in the text, never inferred from the gold label.
    """
    from .evals import autopsy as ap
    from .evals.goldenset import load

    cases = load(path)
    if case_id:
        cases = [c for c in cases if c.case_id == case_id]
    if not cases:
        console.print(f"[red]no cases in {path}[/red]")
        raise typer.Exit(1)
    case = cases[0]

    with console.status(f"dissecting {case.case_id}..."):
        report = ap.autopsy(
            case,
            top_n=top_n,
            use_rerank=rerank,
            item_boost=item_boost,
            context_max_chars=context_max_chars,
            context_packing=context_packing,
            facts=ap.parse_fact_spec(facts) if facts else None,
            out_path=out,
        )

    if report.error:
        console.print(f"[red]{report.error}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]HITS IN FINAL-PROMPT ORDER[/bold] (config: {json.dumps(report.config)})")
    _table(
        "selected hits",
        [
            {
                "rank": h["prompt_rank"],
                "chunk_id": h["chunk_id"],
                "item": h["item"] or "-",
                "gold": "YES" if h["relevant"] else "",
                "in_prompt": "yes" if h["included_in_prompt"] else "CUT",
                "chars": h["body_chars"],
                "preview": h["body_preview"][:90].replace("\n", " "),
            }
            for h in report.hits
        ],
    )

    console.print(
        f"\n[bold]GOLD EVIDENCE[/bold]  included={report.gold_included}  "
        f"cut by cap={report.gold_excluded_by_cap}  never retrieved={report.gold_never_retrieved}"
    )
    _table("gold link quality", report.gold_link_quality)

    console.print("\n[bold]FACT COVERAGE[/bold] (literal search of the rendered prompt)")
    _table("coverage", report.coverage_rows())

    if show_context:
        console.print("\n[bold]EXACT CONTEXT SENT TO THE MODEL[/bold]")
        console.print(report.context_text)

    console.print("\n[bold]REFERENCE[/bold]\n" + report.reference)
    console.print("\n[bold]GENERATED ANSWER[/bold]\n" + report.answer)
    console.print(
        f"\n[bold]VERDICT[/bold] {'PASS' if report.passed else 'FAIL'}  "
        f"(p_tok={report.prompt_tokens} c_tok={report.completion_tokens}"
        f"/{report.max_completion_tokens})"
    )
    console.print("[bold]JUDGE RATIONALE[/bold]\n" + report.judge_rationale)

    judge_runs = None
    if replay:
        console.print(f"\n[bold]GENERATION REPLAY[/bold] x{replay} (retrieval bypassed)")
        for row in ap.replay_generation(report, replay):
            console.print(json.dumps(row, indent=2)[:1400])
    if judge_replay:
        console.print(f"\n[bold]JUDGE REPLAY[/bold] x{judge_replay} (answer held fixed)")
        judge_runs = ap.replay_judge(
            report, repeats=judge_replay, judge_model=judge_model or None
        )
        for row in judge_runs:
            console.print(json.dumps(row, indent=2)[:900])

    console.print(f"\n[bold]{ap.diagnose(report, judge_runs)}[/bold]")
    console.print(f"[dim]full report: {out}[/dim]")


@eval_app.command("judge-ab")
def eval_judge_ab(
    pairs_path: str = typer.Option("", help="Frozen pairs JSON. Empty = build from --from-autopsy."),
    from_autopsy: str = typer.Option("", help="Autopsy report to freeze a pair from."),
    contracts: str = typer.Option("v1,v2", help="Judge contracts to compare."),
    models: str = typer.Option("", help="Judge models. Empty = the configured one only."),
    repeats: int = typer.Option(5),
    controls: bool = typer.Option(True, help="Add negative controls derived from each good answer."),
    save_pairs_to: str = typer.Option("evalset/judge_pairs.json"),
    out: str = typer.Option("reports/judge_ab.json"),
) -> None:
    """Grade frozen pairs across judge contracts and models, one variable per comparison.

    Retrieval and generation are held out entirely: every replay grades identical
    bytes, so the only things that move are the rubric and the model.

    Changing both at once proves nothing -- if v1+mini fails and v2+gpt-4o
    passes, the experiment has not said which caused it. Only cells differing in
    exactly one dimension are reported as conclusions.

    Negative controls are not optional. Replaying one known-good answer until it
    passes measures whether a rubric is permissive, not whether it is correct, so
    each good answer is mutated into a contradiction, an omission, a hedge and a
    fabrication. A contract has to pass the good answer AND fail those.
    """
    from .evals import judge_lab as lab

    if pairs_path:
        pairs = lab.load_pairs(pairs_path)
    elif from_autopsy:
        pairs = [lab.freeze_from_autopsy(from_autopsy)]
    else:
        console.print("[red]pass --pairs-path or --from-autopsy[/red]")
        raise typer.Exit(1)

    if controls:
        expanded: list = []
        for pair in pairs:
            expanded.extend(lab.build_controls(pair) if pair.label == "good" else [pair])
        pairs = expanded

    lab.save_pairs(pairs, save_pairs_to)
    console.print(f"[dim]{len(pairs)} frozen pairs -> {save_pairs_to}[/dim]")
    for p in pairs:
        console.print(
            f"  {p.pair_id[:52]:52} expect={'PASS' if p.expected_verdict else 'FAIL'}  {p.label}"
        )

    contract_list = tuple(c.strip() for c in contracts.split(",") if c.strip())
    model_list = tuple(m.strip() for m in models.split(",") if m.strip())

    def progress(run) -> None:
        if run.error:
            console.print(f"  [red]{run.contract}/{run.model} {run.pair_id[:40]}: {run.error[:70]}[/red]")

    with console.status("judging..."):
        runs = lab.run_grid(
            pairs,
            contracts=contract_list,
            models=model_list,
            repeats=repeats,
            progress=progress,
        )

    cells = lab.summarise(runs)
    payload = lab.save_runs(runs, cells, out)
    console.print()
    _table("judge cells", [c.row() for c in cells])
    if payload["single_variable_comparisons"]:
        _table("single-variable comparisons", payload["single_variable_comparisons"])
    else:
        console.print("[yellow]no single-variable comparison available in this grid[/yellow]")
    console.print(f"\n[bold]{payload['verdict']}[/bold]")
    console.print(f"[dim]full report: {out}[/dim]")


@eval_app.command("judge-kappa")
def eval_judge_kappa(
    run_key: str = typer.Argument(..., help="Run whose human-labelled narrative answers to grade."),
    contracts: str = typer.Option("v2,v3", help="Contracts to compare, baseline first."),
    models: str = typer.Option("", help="Judge models. Empty = the configured one."),
    repeats: int = typer.Option(
        # Kept in step with judge_lab.CALIBRATION_REPEATS by a test. Written as a
        # literal because a typer default is evaluated at import time and cli.py
        # imports the evals package lazily on purpose.
        9,
        help="Repeats per pair; verdicts taken by majority. Nine because judge "
        "non-determinism at temperature 0 was as large as the effects measured.",
    ),
    structures: str = typer.Option(
        "current", help="Prompt layouts to compare, e.g. current,reference_last"
    ),
    duplicate: str = typer.Option(
        "", help="Run this cell twice as a noise control, e.g. v3_1/reference_last"
    ),
    out: str = typer.Option("reports/judge_kappa.json"),
) -> None:
    """Compare judge contracts against HUMAN labels, and report kappa per contract.

    The human labels are the bar. Mutation controls measure whether a judge
    catches breakage it can see; `reports/judge_v2_with_omission.json` scored
    omission 8/8 while real omissions slipped past the same judge seven times on
    the same run. Truncating an answer makes an obviously stunted one; a real
    omission is fluent and covers part of a multi-part disclosure.

    Reports kappa with a bootstrap interval, the confusion matrix, and the
    per-case flips between contracts. On 24 labels a point estimate crossing
    0.60 is not decisive on its own -- the false-positive count and which
    specific cases changed are what distinguish a real fix from a lucky one.
    """
    from .evals import judge_lab as lab

    pairs = lab.pairs_from_labels(run_key)
    if not pairs:
        console.print(f"[red]no human-labelled narrative answers for {run_key}[/red]")
        console.print("[dim]run: edgar-intel eval label <run_key> --n 40[/dim]")
        raise typer.Exit(1)

    comp = lab.composition(pairs)
    _table("calibration set composition", [
        {"metric": "total labelled pairs", "value": comp["total"]},
        {"metric": "human PASS", "value": comp["human_pass"]},
        {"metric": "human FAIL", "value": comp["human_fail"]},
        {"metric": "by kind", "value": json.dumps(comp["by_kind"])},
        {"metric": "by ticker", "value": json.dumps(comp["by_ticker"])},
    ])
    for warning in comp["warnings"]:
        console.print(f"[yellow]{warning}[/yellow]")

    contract_list = tuple(c.strip() for c in contracts.split(",") if c.strip())
    model_list = tuple(m.strip() for m in models.split(",") if m.strip())
    structure_list = tuple(s.strip() for s in structures.split(",") if s.strip())

    def progress(run) -> None:
        if run.error:
            console.print(f"  [red]{run.contract} {run.pair_id[:40]}: {run.error[:70]}[/red]")

    with console.status("judging..."):
        runs = lab.run_grid(
            pairs, contracts=contract_list, models=model_list,
            repeats=repeats, structures=structure_list,
            duplicate=duplicate or None, progress=progress,
        )

    cells = lab.summarise(runs)
    payload = lab.save_runs(runs, cells, out)
    console.print()

    # Headline kappa is per labelled pair, by majority over repeats. Counting
    # each repeat as its own observation multiplies n by the repeat count and
    # narrows the interval without adding information -- the unit a human
    # labelled is the answer, not one measurement of it.
    by_pair = lab.confusion_by_pair(runs)
    _table("kappa against human labels (per pair, majority of repeats)",
           [c.row() for c in by_pair])
    unstable = {p for c in by_pair for p in c.unstable_pairs}
    if unstable:
        console.print(
            f"[yellow]{len(unstable)} pair(s) where repeats disagreed: "
            f"{', '.join(sorted(unstable))}[/yellow]"
        )
    payload["by_pair"] = [c.row() for c in by_pair]
    payload["composition"] = comp
    payload["experiment"] = {
        "run_key": run_key,
        "contracts": list(contract_list),
        "structures": list(structure_list),
        "models": list(model_list) or ["<configured>"],
        "repeats": repeats,
        "pair_count": len(pairs),
        "duplicate_cell": duplicate or None,
        "context_supplied": any(bool(p.source_context) for p in pairs),
    }

    # The noise floor, printed before any comparison, because a delta is only
    # readable against it.
    variance = lab.duplicate_variance(by_pair)
    if variance:
        _table("duplicate control: identical conditions run twice", variance)
        payload["duplicate_variance"] = variance
    else:
        console.print(
            "[yellow]no duplicate-control cell: pass --duplicate <contract>/<structure>, "
            "or no delta can be separated from judge noise[/yellow]"
        )

    # Compare whatever the grid actually varied.
    if len(structure_list) >= 2:
        baseline = f"{contract_list[-1]}/{structure_list[0]}"
        candidate = f"{contract_list[-1]}/{structure_list[-1]}"
    elif len(contract_list) >= 2:
        baseline, candidate = contract_list[0], contract_list[-1]
    else:
        baseline = candidate = ""

    if baseline and candidate:
        model = model_list[0] if model_list else next((r.model for r in runs), "")
        flip_rows = lab.flips(runs, baseline, candidate, model)
        if flip_rows:
            _table(f"per-case flips: {baseline} -> {candidate}", flip_rows)
        else:
            console.print(
                f"[yellow]no case changed verdict between {baseline} and {candidate}"
                "[/yellow]"
            )

        ok, message = lab.regression_guard(flip_rows)
        console.print(f"[{'green' if ok else 'red'}]{message}[/{'green' if ok else 'red'}]")
        payload["flips"] = flip_rows
        payload["regression_guard"] = message

        # The declared gate, checked clause by clause. An aggregate can be
        # satisfied by the wrong cases: v3 reached kappa 0.6667 by fixing three
        # false positives and breaking two correct answers. When a duplicate
        # control ran, beating its spread is a clause too.
        adoption = lab.check_adoption(
            runs,
            candidate=candidate,
            baseline=baseline,
            model=model,
            variance_rows=variance if variance else None,
        )
        _table(f"adoption criteria: {adoption['candidate']}", adoption["clauses"])
        colour = "green" if adoption["adopted"] else "red"
        console.print(f"[{colour}]{adoption['summary']}[/{colour}]")
        payload["adoption"] = adoption

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    console.print(f"\n[bold]{payload['verdict']}[/bold]")
    console.print(f"[dim]full report: {out}[/dim]")


@eval_app.command("judge-targets")
def eval_judge_targets(
    run_key: str = typer.Argument(..., help="Run whose narrative answers to label."),
    want: int = typer.Option(40, help="How many to list; 40 is the stability floor."),
) -> None:
    """Which narrative cases to hand-label next, ordered for diversity.

    Labelling is the expensive step and it is human, so the order matters. This
    puts the case whose (kind, ticker) is least represented in the existing
    labels first, so a set of 40 does not turn out to be 30 legal questions about
    two companies.

    It never suggests a verdict. The judge's own opinion is shown for context but
    is deliberately not a tiebreak -- a calibration set chosen to agree with the
    judge cannot measure the judge.
    """
    from .evals import judge_lab as lab

    targets = lab.label_targets(run_key, want)
    if not targets:
        console.print(f"[green]every narrative answer in {run_key} is already labelled[/green]")
        return
    _table(f"label these next ({len(targets)})", targets)
    console.print(f"[dim]then: edgar-intel eval label {run_key} --n {len(targets)}[/dim]")


@eval_app.command("retrieval")
def eval_retrieval(
    path: str = typer.Option("evalset/golden.json"),
    limit: int = typer.Option(0, help="Use only the first N cases."),
    out: str = typer.Option("reports/retrieval_sweep.json"),
) -> None:
    """Sweep retrieval configurations and attribute where the evidence is lost.

    Calls no LLM, so it is free and fast. Run it before tuning anything: the
    difference between "the evidence was never retrieved" and "the reranker
    buried it" decides whether you touch chunking or ranking, and the two fixes
    have nothing in common.
    """
    from .evals.goldenset import load
    from .evals.retrieval_eval import sweep

    cases = load(path)
    if limit:
        cases = cases[:limit]

    payload = sweep(cases, out_path=out, progress=lambda msg: console.log(msg))
    _table(f"retrieval sweep ({payload['n_cases']} gradeable cases)", payload["rows"])
    console.print()
    console.print(f"[bold]{payload['diagnosis']}[/bold]")
    console.print(f"[dim]full report: {out}[/dim]")


@eval_app.command("rerank-check")
def eval_rerank_check(
    path: str = typer.Option("evalset/golden.json"),
    sample: int = typer.Option(30, help="Queries to sample."),
) -> None:
    """Measure how much of each passage the cross-encoder actually reads.

    Run this when the sweep shows reranking making results worse rather than
    better. A cross-encoder that ranks correct passages down is usually not a
    bad model -- it is a model shown only the first part of the passage, with
    the answer past its token limit.
    """
    import statistics

    from .evals.goldenset import load
    from .providers import get_reranker
    from .retrieval.search import search

    reranker = get_reranker()
    if not hasattr(reranker, "truncation_report"):
        console.print("[yellow]Active reranker has no tokenizer to inspect "
                      "(EDGAR_EMBED_PROVIDER=fake?). Set it to 'local'.[/yellow]")
        raise typer.Exit(1)

    cases = [c for c in load(path) if c.relevant_chunk_ids][:sample]
    rows, seen_pcts, trunc_pcts = [], [], []

    with console.status("scoring sampled queries..."):
        for case in cases:
            result = search(
                case.question, use_rerank=False, top_n=50,
                ticker=case.ticker, fiscal_year=case.fiscal_year,
            )
            if not result.hits:
                continue
            rep = reranker.truncation_report(case.question, [h.body for h in result.hits])
            rows.append(rep)
            seen_pcts.append(rep["median_seen_pct"])
            trunc_pcts.append(rep["truncated_pct"])

    if not rows:
        console.print("[red]no passages retrieved; cannot assess[/red]")
        raise typer.Exit(1)

    median_seen = statistics.median(seen_pcts)
    median_trunc = statistics.median(trunc_pcts)
    budget = statistics.median(r["passage_budget_tokens"] for r in rows)
    med_tokens = statistics.median(r["median_passage_tokens"] for r in rows)
    max_tokens = max(r["max_passage_tokens"] for r in rows)

    _table("cross-encoder input budget", [{
        "queries sampled": len(rows),
        "passage budget (tokens)": int(budget),
        "median passage (tokens)": int(med_tokens),
        "longest passage (tokens)": int(max_tokens),
        "passages truncated %": round(median_trunc, 1),
        "median passage seen %": round(median_seen, 1),
    }])

    console.print()
    if median_trunc >= 50:
        console.print(
            f"[bold red]CONFIRMED: {median_trunc:.0f}% of passages are truncated; "
            f"the model reads about {median_seen:.0f}% of a typical chunk.[/bold red]\n"
            "It is scoring passages it has only partly read, so a figure in the tail "
            "of a chunk is invisible and the correct passage gets ranked down.\n"
            "Fix by making chunks fit the scorer, not by changing the model: lower "
            "--target-tokens on `index build` (the 4-chars-per-token estimate badly "
            "overshoots on financial text), or rerank a centred window rather than "
            "the whole chunk."
        )
    elif median_trunc >= 15:
        console.print(
            f"[bold yellow]PARTIAL: {median_trunc:.0f}% of passages are truncated.[/bold yellow]\n"
            "Enough to hurt the long chunks but not enough to explain a large drop "
            "on its own. Worth fixing, but keep looking."
        )
    else:
        console.print(
            f"[bold]NOT the cause: only {median_trunc:.0f}% of passages are "
            f"truncated.[/bold]\n"
            "The cross-encoder is reading essentially the whole passage, so poor "
            "reranking is a domain-fit problem -- ms-marco is trained on web "
            "prose, and filing chunks are pipe-separated numeric tables. Drop it "
            "and report the measurement, or try a reranker trained on tabular or "
            "financial text."
        )


@eval_app.command("failures")
def eval_failures(run_key: str = typer.Argument(...)) -> None:
    """Split failures into retrieval misses versus generation misses."""
    from .evals.report import failure_breakdown

    console.print_json(json.dumps(failure_breakdown(run_key), indent=2))


@eval_app.command("label")
def eval_label(
    run_key: str = typer.Argument(...),
    n: int = typer.Option(20, help="How many answers to label this session."),
) -> None:
    """Hand-label judged answers so the judge can be calibrated.

    This is the unglamorous half hour that makes every narrative number in the
    report meaningful. Without it, kappa cannot be computed and the judge's
    pass rate is an unverified claim.
    """
    from .evals.judge import calibration_pairs, compute_kappa, kappa_verdict, record_human_label, sample_for_labelling

    run = db.query_one("SELECT id FROM eval_runs WHERE run_key = %s", (run_key,))
    if not run:
        console.print(f"[red]no run named {run_key}[/red]")
        raise typer.Exit(1)

    rows = sample_for_labelling(run["id"], n)
    if not rows:
        console.print("[yellow]no narrative results to label[/yellow]")
        return

    for i, row in enumerate(rows, start=1):
        console.rule(f"{i}/{len(rows)}  {row['case_id']}")
        console.print(f"[bold]Expected:[/bold] {row['expected']}")
        console.print(f"[bold]Answer:[/bold]   {row['answer']}")
        console.print(f"[dim]judge said: {row['passed']} -- {row['judge_rationale']}[/dim]")
        choice = typer.prompt("Is the judge's verdict correct? (y/n/s to skip)", default="s")

        if choice.lower().startswith("s"):
            continue

        judge_verdict = bool(row["passed"])
        agrees_with_judge = choice.lower().startswith("y")

        human_verdict = (
            judge_verdict
            if agrees_with_judge
            else not judge_verdict
        )

        record_human_label(
            row["case_id"],
            row["answer"],
            human_verdict,
        )

    kappa = compute_kappa(calibration_pairs(run["id"]))
    console.print(kappa_verdict(kappa))


# -------------------------------------------------------------------- agent
@agent_app.command("ask")
def agent_ask(
    question: str = typer.Argument(...),
    max_steps: int = typer.Option(0),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    from .agent.loop import run_agent

    run = run_agent(question, max_steps=max_steps or None)
    if json_out:
        console.print_json(json.dumps(run.as_dict(), indent=2, default=str))
    else:
        console.print(run.render())


@agent_app.command("stats")
def agent_stats_cmd() -> None:
    from .agent.loop import outcome_stats

    stats = outcome_stats()
    _table(f"agent outcomes (last {stats['total']})", stats["by_outcome"])


# ----------------------------------------------------------------- finetune
@ft_app.command("dataset")
def ft_dataset(
    out_dir: str = typer.Option("data/finetune"),
    strategy: str = typer.Option("section_aware"),
) -> None:
    from .finetune.dataset import build_dataset

    console.print_json(json.dumps(build_dataset(out_dir, strategy), indent=2))


@ft_app.command("train")
def ft_train(
    base_model: str = typer.Option("meta-llama/Llama-3.1-8B-Instruct"),
    train_file: str = typer.Option("data/finetune/train.jsonl"),
    output_dir: str = typer.Option("artifacts/lora-extract"),
    epochs: float = typer.Option(2.0),
) -> None:
    from .finetune.train_lora import TrainConfig, train

    manifest = train(TrainConfig(
        base_model=base_model, train_file=train_file,
        output_dir=output_dir, epochs=epochs,
    ))
    console.print_json(json.dumps(manifest, indent=2))


@ft_app.command("benchmark")
def ft_benchmark(
    held_out: str = typer.Option("data/finetune/held_out.jsonl"),
    limit: int = typer.Option(100),
) -> None:
    from .finetune.benchmark import compare

    payload = compare(held_out, limit)
    _table("fine-tune vs prompting", payload["arms"])
    console.print(f"\n[bold]{payload['interpretation']}[/bold]")


# -------------------------------------------------------------------- bench
@bench_app.command("run")
def bench_run(
    base_url: str = typer.Option("http://localhost:8000"),
    endpoint: str = typer.Option("/search"),
    requests: int = typer.Option(200),
    concurrency: int = typer.Option(8),
) -> None:
    from .serving.bench import run_bench

    console.print(run_bench(base_url, endpoint, requests, concurrency).render())


@bench_app.command("sweep")
def bench_sweep(
    base_url: str = typer.Option("http://localhost:8000"),
    endpoint: str = typer.Option("/search"),
    requests_per_level: int = typer.Option(100),
) -> None:
    from .serving.bench import sweep

    payload = sweep(base_url, endpoint, requests_per_level=requests_per_level)
    _table("concurrency sweep", payload["levels"])
    console.print(f"\n[bold]{payload['interpretation']}[/bold]")


# ---------------------------------------------------------------------- mcp
@mcp_app.command("serve")
def mcp_serve() -> None:
    from .agent.mcp_server import serve_stdio

    serve_stdio()


@mcp_app.command("config")
def mcp_config() -> None:
    import os

    from .agent.mcp_server import client_config

    console.print_json(json.dumps(client_config(cwd=os.getcwd()), indent=2))


# ------------------------------------------------------------------ fixture
@fixture_app.command("load")
def fixture_load(path: str = typer.Option("tests/fixtures/mini_corpus.json")) -> None:
    from .ingest.pipeline import load_fixture

    console.print_json(json.dumps(load_fixture(path).as_dict(), indent=2))


@app.command("config")
def show_config() -> None:
    s = get_settings()
    redacted = s.model_dump()
    if redacted.get("llm_api_key"):
        redacted["llm_api_key"] = "***"
    console.print_json(json.dumps(redacted, indent=2, default=str))


if __name__ == "__main__":
    app()
