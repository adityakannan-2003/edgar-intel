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
def init_db(schema: str = "sql/001_schema.sql") -> None:
    """Apply the database schema."""
    db.apply_schema(schema)
    console.print("[green]schema applied[/green]")


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


@eval_app.command("run")
def eval_run(
    path: str = typer.Option("evalset/golden.json"),
    label: str = typer.Option(""),
    strategy: str = typer.Option(""),
    mode: str = typer.Option("hybrid"),
    rerank: bool = typer.Option(True),
    limit: int = typer.Option(0, help="Run only the first N cases."),
    git_sha: str = typer.Option(""),
) -> None:
    from .evals.goldenset import load
    from .evals.runner import run_suite

    cases = load(path)
    if limit:
        cases = cases[:limit]

    def progress(i: int, total: int, result) -> None:
        mark = "." if result.passed else "F"
        console.print(mark, end="")
        if i % 50 == 0 or i == total:
            console.print(f" {i}/{total}")

    _, summary = run_suite(
        cases, label=label, strategy=strategy or None, mode=mode,
        use_rerank=rerank, sha=git_sha, progress=progress,
    )
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
        choice = typer.prompt("Correct? (y/n/s to skip)", default="s")
        if choice.lower().startswith("s"):
            continue
        record_human_label(row["case_id"], row["answer"], choice.lower().startswith("y"))

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
