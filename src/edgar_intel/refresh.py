"""Scheduled data refresh: new 10-Ks from EDGAR into the live corpus, only if they pass.

    EDGAR_DB_DSN=<empty staging db> REFRESH_LIVE_DSN=<live db> edgar-intel refresh run

The live database is read throughout and written once, at the end, in one
transaction. Until then everything happens in a staging copy:

1. find     each company's newest 10-Ks on EDGAR, against what live holds.
            Nothing newer: report "nothing new" and stop, before any copy.
2. stage    copy the live corpus into the empty staging database.
3. ingest   only the new filings: sections, chunks, embeddings, XBRL facts.
            `ingest run` is not reused because it re-parses every filing, and
            that cascades through `chunks.section_id`, which moves every chunk id
            the golden set points at.
4. check    parse (the doctor's eight checks), XBRL integrity (verify-facts'
            four, per company), every headline figure found in its own filing's
            text, and chunks embedded and of plausible count. Then the eval gate:
            the new questions answered by the shipped pipeline, and every
            existing question retrieving what it retrieved before.
5. publish  every staging row above the snapshot's id high-water mark, the new
            golden cases and the run's row, in one live transaction that first
            checks live has not moved since the snapshot.

A failure at 4 stops 5, and the report names the filing and the check.
docs/REFRESH.md explains each choice.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from . import db
from .config import get_settings
from .evals.schemas import EvalCase

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
CORPUS = ("companies", "filings", "sections", "chunks", "xbrl_facts")  # foreign-key order
SERIAL = ("filings", "sections", "chunks", "xbrl_facts")
# The density of the original set: 20 single-hop cases per company over three
# filings, and 6 comparisons over two pairs of years.
SINGLE_PER_FILING, YOY_PER_FILING = 7, 3
RETRIEVAL_TOLERANCE = 0.03  # the CI eval gate's tolerance


class RefreshError(RuntimeError):
    """A precondition failed; nothing was published."""


@dataclass(slots=True)
class Check:
    subject: str  # "AAPL 0000320193-26-000026 FY2026", or "eval gate"
    name: str
    passed: bool
    detail: str


def _git_sha() -> str:
    """This code's commit, "-dirty" if edited: not the shell's, which `evals.runner.git_sha` reads."""
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:7]
    try:
        return subprocess.check_output(["git", "-C", str(SQL_DIR.parent), "describe", "--always", "--dirty",
                                        "--abbrev=7"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


# ------------------------------------------------------------------ copying
def _columns(live: psycopg.Connection, stage: psycopg.Connection, table: str) -> str:
    """Columns both databases have, generated ones excluded (COPY cannot write them)."""
    sql = """SELECT column_name FROM information_schema.columns
              WHERE table_schema = 'public' AND table_name = %s AND is_generated = 'NEVER'
              ORDER BY ordinal_position"""
    theirs = {r["column_name"] for r in stage.execute(sql, (table,))}
    return ", ".join(r["column_name"] for r in live.execute(sql, (table,)) if r["column_name"] in theirs)


def _copy(src: psycopg.Connection, dst: psycopg.Connection, table: str, cols: str,
          where: str = "TRUE") -> int:
    with src.cursor().copy(f"COPY (SELECT {cols} FROM {table} WHERE {where} ORDER BY 1) TO STDOUT") as out, \
            dst.cursor() as cur:
        with cur.copy(f"COPY {table} ({cols}) FROM STDIN") as inp:
            for block in out:
                inp.write(block)
        return cur.rowcount


def high_water(conn: psycopg.Connection) -> dict[str, int]:
    return {t: conn.execute(f"SELECT COALESCE(MAX(id), 0) AS n FROM {t}").fetchone()["n"] for t in SERIAL}


def _bump_sequences(conn: psycopg.Connection) -> None:
    """Rows copied with their ids leave the sequences behind; catch them up."""
    for t in SERIAL:
        seq = conn.execute("SELECT pg_get_serial_sequence(%s, 'id') AS s", (t,)).fetchone()["s"]
        conn.execute(f"SELECT setval('{seq}', GREATEST((SELECT COALESCE(MAX(id), 1) FROM {t}),"
                     f" (SELECT last_value FROM {seq})))")


def _table_exists(conn: psycopg.Connection, table: str) -> bool:
    return conn.execute("SELECT to_regclass(%s) AS t", (f"public.{table}",)).fetchone()["t"] is not None


def stage_copy(live: psycopg.Connection, stage: psycopg.Connection) -> dict[str, int]:
    """Copy the live corpus into staging. Staging must be empty: never a second live."""
    if _table_exists(stage, "filings") and stage.execute("SELECT EXISTS (SELECT 1 FROM filings) AS e").fetchone()["e"]:
        raise RefreshError("the staging database (EDGAR_DB_DSN) already holds filings. It must be an "
                           "empty database the job may overwrite, never the live one.")
    db.apply_schema(str(SQL_DIR))
    copied = {t: _copy(live, stage, t, _columns(live, stage, t)) for t in CORPUS}
    _bump_sequences(stage)
    # A bulk-loaded table has no statistics until autovacuum gets to it, and the
    # planner's guesses shift as rows arrive: on 11 Nov that reordered tied lexical
    # scores for 93 of 232 questions between "before" and "after" with no new filing
    # involved. Analyzed, both measurements run on settled plans, as live does.
    stage.execute("ANALYZE")
    return copied


# -------------------------------------------------------------------- find
def _label(row: dict[str, Any]) -> str:
    return f"{row['ticker']} {row['accession']} FY{row['fiscal_year'] or '?'}"


def find_new(live: psycopg.Connection, client, tickers: list[str] | None = None) -> list[dict[str, Any]]:
    """10-Ks filed after the newest one live holds for each company it serves.

    Companies come from live, not from the configured universe: adding a company
    is a corpus decision, not a refresh. Older filings that live lacks are not
    backfilled either; the refresh only moves the corpus forward.
    """
    from .ingest.edgar_client import recent_filings
    from .ingest.xbrl import fiscal_year_ending

    s = get_settings()
    wanted = {t.upper() for t in tickers} if tickers else None
    companies = live.execute(
        """SELECT co.cik, co.ticker, co.name, MAX(f.filing_date) AS latest,
                  COALESCE(array_agg(f.accession) FILTER (WHERE f.accession IS NOT NULL), '{}') AS held
             FROM companies co LEFT JOIN filings f ON f.cik = co.cik
            GROUP BY 1, 2, 3 ORDER BY 2"""
    ).fetchall()
    new = []
    for co in companies:
        if wanted and (co["ticker"] or "").upper() not in wanted:
            continue
        rows = recent_filings(client.submissions(co["cik"]), s.forms, limit=s.years_back,
                              fetch_page=client.submissions_page)
        for r in rows:
            filed = date.fromisoformat(r["filing_date"])
            if r["accession"] in co["held"] or (co["latest"] and filed <= co["latest"]):
                continue
            fy = fiscal_year_ending(date.fromisoformat(r["period_end"])) if r.get("period_end") else None
            new.append({**r, "cik": co["cik"], "ticker": co["ticker"], "name": co["name"], "fiscal_year": fy})
    return sorted(new, key=lambda r: (r["ticker"], r["filing_date"]))


# ------------------------------------------------------------------ ingest
_INSERT_CHUNK = """INSERT INTO chunks (filing_id, section_id, strategy, ordinal, token_est, body)
                   VALUES (%s, %s, %s, %s, %s, %s)"""


def ingest_new(client, new: list[dict[str, Any]], progress=print) -> tuple[dict[str, int], list[Check]]:
    """Fetch, parse, chunk and embed the new filings into staging, and their facts.

    Returns each accession's staging filing id, and the doctor's parse checks.
    A fetch that fails after retries raises (the run fails); a document that
    will not parse is a failed check (the run is blocked).
    """
    from .chunking import estimate_tokens
    from .ingest.doctor import diagnose_html
    from .ingest.parse import parse_filing
    from .ingest.pipeline import replace_sections, upsert_facts, upsert_filing
    from .ingest.xbrl import extract_facts
    from .retrieval.index import build_chunks, embed_pending

    s = get_settings()
    ids: dict[str, int] = {}
    checks: list[Check] = []
    for row in new:
        progress(f"ingest {_label(row)}")
        url = client.filing_url(row["cik"], row["accession"], row["primary_doc"])
        filing_id = upsert_filing(row["cik"], row, url)
        ids[row["accession"]] = filing_id
        html = client.filing_document(row["cik"], row["accession"], row["primary_doc"])
        try:
            diag = diagnose_html(row["ticker"], row["accession"], row["fiscal_year"], html)
            replace_sections(filing_id, parse_filing(html))
        except Exception as exc:  # bad markup is bad data, not a crash
            checks.append(Check(_label(row), "filing parses", False, f"{type(exc).__name__}: {exc}"))
            continue
        checks += [Check(_label(row), c.name, c.passed, c.detail) for c in diag.checks]
        # The defaults `index build` uses; see docs/REFRESH.md on the served index's older chunker.
        chunks = build_chunks(filing_id, s.default_strategy, 512, 64)
        db.execute_many(_INSERT_CHUNK, [
            (filing_id, c.section_id, s.default_strategy, c.ordinal, estimate_tokens(c.body), c.body)
            for c in chunks
        ])
    for cik in sorted({r["cik"] for r in new}):
        progress(f"companyfacts {cik}")
        upsert_facts(extract_facts(cik, client.company_facts(cik), forms=tuple(s.forms)))
    embed_pending(s.default_strategy, batch=s.embed_batch, progress=progress)
    return ids, checks


# ------------------------------------------------------------------ checks
def figure_needles(value: float, unit: str) -> list[str]:
    """How a statement prints `value`: exactly, at a scale it divides evenly.

    Stricter than the golden set's evidence linker, which also tries rounded
    scales: "4,162" for a corrupted 4,161,610,000,000 turns up somewhere in any
    10-K, and a check that a wrong figure passes is no check. A round value
    (924,000,000 shares; $2.0 billion) has no form specific enough to look for,
    and gets no needles: it is reported as not checkable rather than as missing.
    """
    from .evals.goldenset import _decimal_form

    v = abs(value)
    if unit.startswith("USD/") or v < 1_000:
        return [_decimal_form(v)]
    if len(f"{v:.0f}".rstrip("0")) < 4:
        return []
    exact = [f"{v / s:,.0f}" for s in (1, 1_000, 1_000_000) if v % s == 0]
    tenths = [f"{v / s:,.1f}" for s in (1_000_000, 1_000_000_000) if v % s and not v % (s // 10)]  # "499.4"
    return [n for n in exact + tenths if sum(ch.isdigit() for ch in n) >= 4]


def _core_facts(cik: str, fiscal_year: int) -> list[dict[str, Any]]:
    from .evals.goldenset import _canonicalize_fact_rows
    from .ingest.xbrl import CORE_TAGS

    rows = db.query(
        """SELECT cik, tag, unit, fiscal_year, value::float8 AS value, prior_year_value
             FROM xbrl_facts WHERE cik = %s AND fiscal_year = %s AND fiscal_period = 'FY'
            ORDER BY tag""",
        (cik, fiscal_year),
    )
    return _canonicalize_fact_rows([r for r in rows if r["tag"] in CORE_TAGS])


def filing_checks(row: dict[str, Any], filing_id: int) -> list[Check]:
    """Figures and chunks of one new filing, in staging."""
    from .ingest.xbrl import CORE_TAGS, format_value

    s = get_settings()
    label = _label(row)
    if row["fiscal_year"] is None:
        return [Check(label, "fiscal year known", False, "EDGAR gave no period end, so no fiscal year")]

    facts = _core_facts(row["cik"], row["fiscal_year"])
    # Last year's filing is the spec: filers differ (CAT tags no NetIncomeLoss at all),
    # and a fact a company reported last year and not this year was lost on the way in.
    last = {f["tag"] for f in _core_facts(row["cik"], row["fiscal_year"] - 1)}
    lost = sorted(last - {f["tag"] for f in facts})
    out = [Check(label, "same core XBRL facts as last year", bool(facts) and not lost,
                 f"{len(facts)} core facts for FY{row['fiscal_year']}, {len(last)} for FY{row['fiscal_year'] - 1}"
                 + (f"; MISSING {', '.join(CORE_TAGS[t] + ' (' + t + ')' for t in lost)}" if lost else ""))]

    text = "\n".join(r["body"] for r in db.query("SELECT body FROM sections WHERE filing_id = %s", (filing_id,)))
    checkable = [f for f in facts if figure_needles(f["value"], f["unit"])]
    absent = [f for f in checkable if not any(n in text for n in figure_needles(f["value"], f["unit"]))]
    enough = len(checkable) >= 0.75 * len(facts)
    out.append(Check(
        label, "XBRL figures appear in the filing text", bool(facts) and enough and not absent,
        f"{len(checkable) - len(absent)} of {len(checkable)} found ({len(facts) - len(checkable)} too round to look for"
        + ("" if enough else "; fewer than 3 in 4 checkable") + ")"
        + "".join(f"; NOT FOUND {CORE_TAGS[f['tag']]} {format_value(f['value'], f['unit'])} "
                  f"(XBRL {f['tag']}, looked for {' / '.join(figure_needles(f['value'], f['unit']))})"
                  for f in absent),
    ))

    c = db.query_one(
        """SELECT COUNT(*) AS n, COUNT(embedding) AS embedded,
                  COUNT(*) FILTER (WHERE btrim(body) = '') AS empty
             FROM chunks WHERE filing_id = %s AND strategy = %s""",
        (filing_id, s.default_strategy),
    )
    out.append(Check(label, "chunks embedded", c["n"] > 0 and c["embedded"] == c["n"] and not c["empty"],
                     f"{c['n']} chunks, {c['embedded']} embedded, {c['empty']} empty"))
    prev = db.query_one(
        """SELECT COUNT(*) AS n FROM chunks ch JOIN filings f ON f.id = ch.filing_id
            WHERE f.cik = %s AND f.fiscal_year = %s AND ch.strategy = %s""",
        (row["cik"], row["fiscal_year"] - 1, s.default_strategy),
    )["n"]
    if prev:
        ratio = c["n"] / prev
        out.append(Check(label, "chunk count in line with last year", 0.5 <= ratio <= 2.0,
                         f"{c['n']} vs {prev} for FY{row['fiscal_year'] - 1} ({ratio:.2f}x; allowed 0.5x to 2x)"))
    return out


def fact_checks(cik: str, ticker: str) -> list[Check]:
    """`ingest verify-facts`' four checks, for one company's whole history."""
    from .ingest.xbrl import Fact, misdated_years, missing_years

    subject = f"{ticker} XBRL facts, all years"
    conflicts = db.query(
        """SELECT tag, fiscal_year FROM xbrl_facts WHERE cik = %s AND fiscal_period = 'FY'
            GROUP BY 1, 2 HAVING COUNT(DISTINCT value) > 1 ORDER BY 2, 1""", (cik,))
    drift = db.query(
        """SELECT x.tag, x.fiscal_year, x.period_end AS fact_end, fl.period_end AS filing_end
             FROM xbrl_facts x JOIN filings fl ON fl.cik = x.cik AND fl.fiscal_year = x.fiscal_year
            WHERE x.cik = %s AND x.fiscal_period = 'FY' AND fl.period_end IS NOT NULL
              AND ABS(x.period_end - fl.period_end) > 7 ORDER BY 2, 1""", (cik,))
    facts = [Fact(**r) for r in db.query(
        """SELECT cik, taxonomy, tag, unit, fiscal_year, fiscal_period, period_start, period_end,
                  value::float8 AS value, accession, form
             FROM xbrl_facts WHERE cik = %s AND fiscal_period = 'FY'""", (cik,))]
    misdated, missing = misdated_years(facts), missing_years(facts)

    def few(rows: list, fmt) -> str:
        return "; ".join(fmt(r) for r in rows[:5]) + (f"; and {len(rows) - 5} more" if len(rows) > 5 else "")

    return [
        Check(subject, "one value per tag and year", not conflicts,
              "none conflict" if not conflicts else few(conflicts, lambda r: f"{r['tag']} FY{r['fiscal_year']}")),
        Check(subject, "fact periods agree with the filing index", not drift,
              "all within a week" if not drift else few(
                  drift, lambda r: f"{r['tag']} FY{r['fiscal_year']} ends {r['fact_end']}, filing {r['filing_end']}")),
        Check(subject, "year labels advance with period ends", not misdated,
              "all consistent" if not misdated else few(misdated, lambda r: f"{r[1]} FY{r[3]} {r[4]} -> FY{r[5]} {r[6]}")),
        Check(subject, "no fiscal year skipped", not missing,
              "none skipped" if not missing else few(missing, lambda r: f"FY{r[1]}")),
    ]


# ------------------------------------------------------------------ golden
def new_cases(row: dict[str, Any]) -> list[EvalCase]:
    """Questions from one new filing's own XBRL facts, in the original set's format."""
    from .evals.goldenset import QUESTION_TEMPLATES, _yoy_cases
    from .ingest.xbrl import CORE_TAGS, format_value

    cik, fy, ticker = row["cik"], row["fiscal_year"], row["ticker"]
    rng = random.Random(f"{ticker}-{fy}")  # reproducible per filing, whatever else is in the corpus
    singles = _core_facts(cik, fy)
    rng.shuffle(singles)
    cases = [
        EvalCase(
            case_id=f"num-{ticker}-{f['tag']}-{fy}", kind="numeric",
            question=rng.choice(QUESTION_TEMPLATES["single"]).format(
                company=row["name"], label=CORE_TAGS[f["tag"]], year=fy),
            expected=format_value(f["value"], f["unit"]), expected_value=f["value"], unit=f["unit"],
            ticker=ticker, fiscal_year=fy, tag=f["tag"], difficulty="single_hop",
        )
        for f in singles[:SINGLE_PER_FILING]
    ]
    pair = _core_facts(cik, fy - 1) + _core_facts(cik, fy)
    cases += _yoy_cases(pair, row["name"], ticker, cik, rng, YOY_PER_FILING, {(cik, fy - 1), (cik, fy)})
    return cases


def load_golden(live: psycopg.Connection) -> list[EvalCase] | None:
    if not _table_exists(live, "golden_cases"):
        return None
    rows = live.execute("SELECT body FROM golden_cases ORDER BY ordinal").fetchall()
    return [EvalCase(**r["body"]) for r in rows] or None


def reference_run(live: psycopg.Connection) -> dict[str, Any] | None:
    """The run `/stats/eval` serves: the latest finished retrieve-then-answer run."""
    return live.execute(
        """SELECT id, run_key, summary FROM eval_runs
            WHERE summary IS NOT NULL AND COALESCE(config->>'pipeline', 'rag') = 'rag'
            ORDER BY finished_at DESC NULLS LAST, id DESC LIMIT 1"""
    ).fetchone()


def bootstrap_golden(live: psycopg.Connection, path: str) -> list[EvalCase]:
    """The original set, rebuilt from the staged copy of live before anything is added.

    Only valid while the corpus is the one the reference run graded, so it is
    checked against that run's questions and answers, and refused on any difference.
    """
    from .evals.goldenset import build

    ref = reference_run(live)
    if ref is None:
        raise RefreshError("live holds no evaluation run to check a rebuilt golden set against")
    cases = build(path)
    graded = {r["case_id"]: (r["question"], r["expected"]) for r in live.execute(
        "SELECT case_id, question, expected FROM eval_results WHERE run_id = %s", (ref["id"],))}
    built = {c.case_id: (c.question, c.expected) for c in cases}
    if built != graded:
        differ = sorted(set(built) ^ set(graded)) or sorted(k for k in built if built[k] != graded[k])
        raise RefreshError(
            f"the golden set rebuilt from live ({len(built)} cases) is not the exam {ref['run_key']} "
            f"graded ({len(graded)} cases); first differences: {', '.join(differ[:5])}")
    return cases


# -------------------------------------------------------------------- gate
def _retrieve(cases: list[EvalCase]) -> dict[str, list[str]]:
    """Passages each question gets under the shipped configuration, as `eval run` asks."""
    from .retrieval.search import search

    return {c.case_id: search(c.question, ticker=c.ticker, fiscal_year=c.fiscal_year).ids for c in cases}


def _hit_at_5(cases: list[EvalCase], got: dict[str, list[str]]) -> float:
    labelled = [c for c in cases if c.relevant_chunk_ids]
    hits = sum(bool(set(got[c.case_id][:5]) & set(c.relevant_chunk_ids)) for c in labelled)
    return hits / len(labelled) if labelled else 0.0


def retrieval_check(cases: list[EvalCase], before: dict, after: dict) -> Check:
    """Search filters on the case's company and year, so a new filing cannot enter an
    old question's candidates. HNSW can still move them: pgvector filters after the
    index returns its candidates (docs/DEPLOY.md §2d). Measured, not assumed."""
    changed = [c.case_id for c in cases if before[c.case_id] != after[c.case_id]]
    b, a = _hit_at_5(cases, before), _hit_at_5(cases, after)
    return Check("eval gate", "existing questions retrieve as before", a >= b - RETRIEVAL_TOLERANCE,
                 f"{len(changed)} of {len(cases)} changed passages; hit@5 {b:.4f} -> {a:.4f}"
                 + (f" (allowed drop {RETRIEVAL_TOLERANCE})" if changed else "")
                 + (f"; changed: {', '.join(changed[:5])}" if changed else ""))


def answer_floor(reference: float, n: int) -> float:
    """The reference accuracy less two standard errors at this many cases.

    Ten or twenty new questions cannot be held to a 3-point tolerance: at n=20
    one case is 5 points. Two standard errors is where a sample that size from a
    system as good as the reference would almost never land below.
    """
    return reference - 2 * math.sqrt(reference * (1 - reference) / n)


def answer_check(cases: list[EvalCase], live: psycopg.Connection, sha: str, progress=print) -> Check:
    from .evals.runner import run_suite

    s = get_settings()
    if s.llm_provider == "fake":
        return Check("eval gate", "new questions answered", False,
                     "not run: EDGAR_LLM_PROVIDER is 'fake', and a fake model says nothing about data")
    ref = reference_run(live)
    ref_acc = ref["summary"].get("numeric_accuracy") if ref else None
    if ref_acc is None:
        return Check("eval gate", "new questions answered", False, "live holds no reference run to compare with")
    progress(f"answering {len(cases)} new questions with {s.llm_model}")
    run_id, summary = run_suite(cases, label="refresh", sha=sha)
    acc, n = summary.numeric_accuracy or 0.0, sum(c.kind == "numeric" for c in cases)
    floor = answer_floor(ref_acc, n)
    wrong = db.query("SELECT case_id, expected, answer FROM eval_results WHERE run_id = %s AND NOT passed "
                     "ORDER BY case_id", (run_id,))
    return Check(
        "eval gate", "new questions answered", acc >= floor,
        f"{round(acc * n)}/{n} = {acc:.3f}; floor {floor:.3f} is {ref['run_key']}'s {ref_acc:.4f} less "
        f"2 standard errors at n={n}; ${summary.cost_per_1k_usd * n / 1000:.4f}"
        + "".join(f"; WRONG {w['case_id']}: expected {w['expected']}, got {(w['answer'] or '')[:80]!r}"
                  for w in wrong[:5]),
    )


# ----------------------------------------------------------------- publish
def _ensure_tables(live: psycopg.Connection) -> None:
    live.execute((SQL_DIR / "006_refresh_runs.sql").read_text(encoding="utf-8"))


def _record(conn: psycopg.Connection, run: dict[str, Any]) -> int:
    return conn.execute(
        """INSERT INTO refresh_runs (started_at, finished_at, status, filings_added, trigger,
                                     git_sha, new_filings, report)
           VALUES (%s, now(), %s, %s, %s, %s, %s::jsonb, %s) RETURNING id""",
        (run["started_at"], run["status"], run["filings_added"], run["trigger"], run["git_sha"],
         json.dumps(run["new_filings"], default=str), run["report"]),
    ).fetchone()["id"]


def publish(live: psycopg.Connection, stage: psycopg.Connection, hwm: dict[str, int], run: dict[str, Any],
            base: list[EvalCase] | None, added: list[EvalCase]) -> tuple[dict[str, int], int]:
    """Append staging's new rows to live, in one transaction, or nothing at all."""
    with live.transaction():
        live.execute("LOCK TABLE filings, sections, chunks, xbrl_facts, golden_cases, refresh_runs "
                     "IN SHARE ROW EXCLUSIVE MODE")
        now = high_water(live)
        if now != hwm:
            raise RefreshError(f"live changed while the refresh ran (high-water {hwm} -> {now}); "
                               "nothing was published, run again")
        rows = {t: _copy(stage, live, t, _columns(live, stage, t), f"id > {int(hwm[t])}") for t in SERIAL}
        _bump_sequences(live)
        run_id = _record(live, run)
        start = live.execute("SELECT COALESCE(MAX(ordinal), -1) + 1 AS n FROM golden_cases").fetchone()["n"]
        golden = [(c, None) for c in base or []] + [(c, run_id) for c in added]
        with live.cursor() as cur:
            cur.executemany(
                "INSERT INTO golden_cases (case_id, ordinal, body, refresh_run_id) VALUES (%s, %s, %s::jsonb, %s)",
                [(c.case_id, start + i, json.dumps(c.as_dict()), rid) for i, (c, rid) in enumerate(golden)])
        rows["golden_cases"] = len(golden)
    return rows, run_id


# ------------------------------------------------------------------ report
def render(run: dict[str, Any], checks: list[Check], notes: list[str]) -> str:
    head = {"ok": "OK", "blocked": "BLOCKED", "failed": "FAILED"}[run["status"]]
    if run["status"] == "ok" and run.get("dry_run") and run["new_filings"]:
        head += f": {len(run['new_filings'])} filing(s) passed, not published (dry run)"
    elif run["status"] == "ok":
        head += f": {run['filings_added']} filing(s) published" if run["filings_added"] else ": nothing new"
    lines = [f"# edgar-intel refresh {head}", "",
             f"{run['started_at']:%Y-%m-%d %H:%M} UTC · trigger {run['trigger']} · git {run['git_sha'] or '?'}"
             + (" · dry run" if run.get("dry_run") else ""), ""]
    lines += notes + [""]
    found = run["new_filings"]
    lines += [f"## New 10-Ks on EDGAR: {len(found)}", ""]
    lines += [f"- {_label(r)}, filed {r['filing_date']}" for r in found] or ["- none since the newest filing live holds"]
    failed = [c for c in checks if not c.passed]
    if failed:
        lines += ["", f"## What failed: {len(failed)}", "", "| filing | check | detail |", "|---|---|---|"]
        lines += [f"| {c.subject} | {c.name} | {c.detail} |" for c in failed]
    if checks:
        lines += ["", f"## Every check: {len(checks) - len(failed)} of {len(checks)} passed", "",
                  "| filing | check | result | detail |", "|---|---|---|---|"]
        lines += [f"| {c.subject} | {c.name} | {'ok' if c.passed else 'FAIL'} | {c.detail} |" for c in checks]
    return "\n".join(lines) + "\n"


def _save(report_dir: str, run: dict[str, Any], report: str, payload: dict[str, Any]) -> str:
    os.makedirs(report_dir, exist_ok=True)
    stem = os.path.join(report_dir, f"refresh-{run['started_at']:%Y%m%dT%H%M%SZ}")
    Path(stem + ".md").write_text(report, encoding="utf-8")
    Path(stem + ".json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    Path(report_dir, "latest.md").write_text(report, encoding="utf-8")
    return stem + ".md"


# --------------------------------------------------------------------- run
def run(live_dsn: str, tickers: list[str] | None = None, dry_run: bool = False, trigger: str = "manual",
        report_dir: str = "reports/refresh", progress=print, client=None) -> tuple[str, str, str]:
    """One refresh. Returns (status, report, report path). `client` is for tests and proofs."""
    from .ingest.edgar_client import EdgarClient

    if live_dsn.strip() == get_settings().db_dsn.strip():
        raise RefreshError("REFRESH_LIVE_DSN and EDGAR_DB_DSN are the same database; staging must be separate")
    os.makedirs(report_dir, exist_ok=True)
    rec = {"started_at": datetime.now(UTC), "status": "failed", "filings_added": 0, "trigger": trigger,
           "git_sha": _git_sha(), "new_filings": [], "dry_run": dry_run}
    checks: list[Check] = []
    notes: list[str] = []
    payload: dict[str, Any] = {}
    run_id = None
    client = client or EdgarClient()
    with psycopg.connect(live_dsn, row_factory=dict_row, autocommit=True) as live:
        try:
            if not dry_run:
                _ensure_tables(live)
            rec["new_filings"] = new = find_new(live, client, tickers)
            if not new:
                rec["status"] = "ok"
                notes.append("Nothing to do: live already holds every company's newest 10-K. No staging, and no "
                             + ("writes at all (dry run)." if dry_run else "writes but this run's row in refresh_runs."))
            else:
                rec["status"], run_id = _stage_check_publish(
                    live, client, new, rec, checks, notes, payload, dry_run, report_dir, progress)
        except Exception as exc:
            rec["status"] = "failed"
            notes.append("**The refresh raised before publishing; the live database is unchanged.** "
                         f"{type(exc).__name__}: {exc}")
        finally:
            client.close()
        report = render(rec, checks, notes)
        if run_id is not None:  # published: the row exists, written in the publish transaction
            live.execute("UPDATE refresh_runs SET report = %s WHERE id = %s", (report, run_id))
        elif not dry_run:
            _record(live, {**rec, "report": report})
    payload.update(run=rec, checks=[asdict(c) for c in checks])
    return rec["status"], report, _save(report_dir, rec, report, payload)


def _stage_check_publish(live, client, new, rec, checks, notes, payload, dry_run, report_dir, progress):
    """Steps 2 to 5 for a run that found new filings. Returns (status, published run id or None)."""
    from .evals.goldenset import link_evidence

    with psycopg.connect(get_settings().db_dsn, row_factory=dict_row, autocommit=True) as stage:
        progress("staging a copy of live")
        copied = stage_copy(live, stage)
        hwm = high_water(stage)
        notes.append(f"Staged a copy of live: {', '.join(f'{n:,} {t}' for t, n in copied.items())}.")

        old = load_golden(live)
        base = None
        if old is None:
            progress("golden set: none stored in live yet; rebuilding the original from the staged copy")
            old = base = bootstrap_golden(live, os.path.join(report_dir, "golden_base.json"))
            notes.append(f"Golden set: live had none stored, so the original {len(base)} cases were rebuilt "
                         "from the staged copy and matched the reference run's exam question for question.")
        progress(f"retrieving for {len(old)} existing questions (before)")
        before = _retrieve(old)

        ids, parse = ingest_new(client, new, progress)
        stage.execute("ANALYZE")
        checks += parse
        for row in new:
            if row["accession"] in ids:
                checks += filing_checks(row, ids[row["accession"]])
        for cik, ticker in sorted({(r["cik"], r["ticker"]) for r in new}):
            checks += fact_checks(cik, ticker)

        known = {c.case_id for c in old}
        added = [c for row in new if row["fiscal_year"] for c in new_cases(row) if c.case_id not in known]
        added = link_evidence(added, get_settings().default_strategy)
        payload["new_cases"] = [c.as_dict() for c in added]
        unlinked = [c.case_id for c in added if not c.relevant_chunk_ids]
        checks.append(Check("golden set", "new questions have evidence in their filing", bool(added) and not unlinked,
                            f"{len(added) - len(unlinked)} of {len(added)} linked"
                            + (f"; unlinked: {', '.join(unlinked)}" if unlinked else "")))

        if any(not c.passed for c in checks):
            notes.append("**Blocked by the data checks. The eval gate was not run: it would spend money grading "
                         "data already known to be bad. Nothing was published; the live database is unchanged.**")
            return "blocked", None

        progress(f"retrieving for {len(old)} existing questions (after)")
        after = _retrieve(old)
        payload["retrieval"] = {"before": before, "after": after}  # every question's passage ids, for diagnosis
        checks.append(retrieval_check(old, before, after))
        checks.append(answer_check(added, live, rec["git_sha"], progress))
        if any(not c.passed for c in checks):
            notes.append("**Blocked by the eval gate. Nothing was published; the live database is unchanged.**")
            return "blocked", None
        if dry_run:
            notes.append(f"Dry run: every check passed; {len(new)} filing(s) and {len(added)} questions would be "
                         "published. Nothing was published or recorded.")
            return "ok", None

        rec["status"], rec["filings_added"] = "ok", len(new)
        counts, run_id = publish(live, stage, hwm, {**rec, "report": None}, base, added)
        notes.append("Published in one transaction: " + ", ".join(f"{n:,} {t}" for t, n in counts.items()) + ".")
        return "ok", run_id
