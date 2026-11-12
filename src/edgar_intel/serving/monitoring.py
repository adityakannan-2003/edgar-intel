"""Monitoring: one read-only JSON endpoint and one page that draws it.

`GET /stats/monitoring` answers four questions about the live system, each from
the place its data already lives (docs/MONITORING.md has the why):

  latency    p50/p95 per endpoint, from a window this process keeps in memory;
  cost       per /ask request, from `agent_traces` (every agent run records it);
  eval       score per full-set eval run over time, from `eval_runs`;
  freshness  when the data last changed, and when the refresh job last ran.

`GET /dashboard` is a static page that fetches that JSON every minute.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from .. import db
from .bench import _percentile

router = APIRouter()

# shortcut: per process and in memory, so a deploy or restart empties it and a
# second machine would keep its own; upgrade to a request_log table if latency
# history must survive deploys. The traces in Grafana keep the long history.
_WINDOW = 1000
_ENDPOINTS = ("/ask", "/search", "/mcp")
_since = datetime.now(UTC)
_latency: dict[str, deque[tuple[float, int]]] = {p: deque(maxlen=_WINDOW) for p in _ENDPOINTS}

# The golden set is 232 cases; smoke, pilot and retry runs are far smaller and
# would read as wild swings on a chart of the system's quality.
_MIN_EVAL_CASES = 100


async def record_latency(request: Request, call_next):
    """Time every POST to a monitored endpoint. /mcp is stateless JSON, so the
    response is complete when `call_next` returns, as for /ask and /search."""
    started = time.perf_counter()
    response = await call_next(request)
    window = _latency.get(request.url.path)
    if window is not None and request.method == "POST":
        window.append(((time.perf_counter() - started) * 1000, response.status_code))
    return response


def _pct(values: list[float], q: float) -> float | None:
    # None, not bench's 0.0: an empty window is "no data", never "0 ms".
    return _percentile(sorted(values), q) if values else None


def _latency_stats() -> dict[str, Any]:
    out = {}
    for path, window in _latency.items():
        ok = [ms for ms, status in window if status < 400]
        p50, p95 = _pct(ok, 0.5), _pct(ok, 0.95)
        out[path] = {
            "requests": len(window),
            "ok": len(ok),
            "client_errors": sum(1 for _, s in window if 400 <= s < 500),
            "server_errors": sum(1 for _, s in window if s >= 500),
            "p50_ms": None if p50 is None else round(p50),
            "p95_ms": None if p95 is None else round(p95),
        }
    return {
        "source": f"this process, last {_WINDOW} POSTs per endpoint; 4xx/5xx excluded from p50/p95",
        "since": _since,
        "endpoints": out,
    }


def _cost_stats() -> dict[str, Any]:
    rows = db.query(
        """
        SELECT created_at, outcome, total_latency_ms, total_cost_usd
          FROM agent_traces ORDER BY id DESC LIMIT 500
        """
    )
    costs = [float(r["total_cost_usd"]) for r in rows if r["total_cost_usd"] is not None]
    latencies = [r["total_latency_ms"] for r in rows if r["total_latency_ms"] is not None]
    mean = sum(costs) / len(costs) if costs else None
    return {
        "source": "agent_traces, last 500 /ask runs; /search and /mcp call no paid model",
        "runs": len(rows),
        "mean_usd": mean,
        "p50_usd": _pct(costs, 0.5),
        "p95_usd": _pct(costs, 0.95),
        "per_1k_usd": None if mean is None else mean * 1000,
        "total_usd": sum(costs),
        "p50_latency_ms": _pct(latencies, 0.5),
        "p95_latency_ms": _pct(latencies, 0.95),
        "first_at": rows[-1]["created_at"] if rows else None,
        "last_at": rows[0]["created_at"] if rows else None,
        # No question text and no trace id: /traces/{id} would show a visitor's question.
        "recent": [
            {
                "created_at": r["created_at"],
                "outcome": r["outcome"],
                "latency_ms": r["total_latency_ms"],
                "cost_usd": r["total_cost_usd"],
            }
            for r in rows[:10]
        ],
    }


def _eval_history() -> dict[str, Any]:
    # Only fields measured without the LLM judge. `/stats/eval` withholds
    # narrative_pass_rate and overall_score until kappa clears the floor; here
    # they are never selected at all.
    runs = db.query(
        """
        SELECT run_key, label, git_sha, finished_at, n_cases,
               COALESCE(config->>'pipeline', 'rag')        AS pipeline,
               (summary->>'numeric_accuracy')::float        AS numeric_accuracy,
               (summary->>'abstention_rate')::float         AS abstention_rate,
               (summary->>'hallucination_rate')::float      AS hallucination_rate,
               (summary->'retrieval'->>'hit@5')::float      AS hit_at_5,
               (summary->'retrieval'->>'mrr')::float        AS mrr,
               (summary->>'cost_per_1k_usd')::float         AS cost_per_1k_usd
          FROM eval_runs
         WHERE summary IS NOT NULL AND finished_at IS NOT NULL AND n_cases >= %s
         ORDER BY finished_at
        """,
        (_MIN_EVAL_CASES,),
    )
    return {
        "source": f"eval_runs with at least {_MIN_EVAL_CASES} cases; judge-free fields only",
        "runs": runs,
    }


def _freshness() -> dict[str, Any]:
    data = db.query_one(
        """
        SELECT COUNT(*) AS filings, MAX(created_at) AS newest_ingested_at,
               MAX(filing_date) AS newest_filing_date
          FROM filings
        """
    )
    # The scheduled refresh records each run in `refresh_runs` (docs/MONITORING.md).
    # Until that table exists the panel reports the job as pending.
    if not db.query_one("SELECT to_regclass('public.refresh_runs') AS t")["t"]:
        return {"data": data, "job": {"status": "pending", "last_run": None, "last_ok": None}}
    cols = "started_at, finished_at, status, filings_added"
    last = db.query_one(f"SELECT {cols} FROM refresh_runs ORDER BY started_at DESC LIMIT 1")
    last_ok = db.query_one(
        f"SELECT {cols} FROM refresh_runs WHERE status = 'ok' ORDER BY started_at DESC LIMIT 1"
    )
    return {
        "data": data,
        "job": {"status": "recorded" if last else "no runs yet", "last_run": last, "last_ok": last_ok},
    }


def _database_stats() -> dict[str, Any]:
    try:
        return {"cost": _cost_stats(), "eval": _eval_history(), "freshness": _freshness()}
    except Exception as exc:
        # The type only: a driver's message can carry the database host.
        return {"database_error": type(exc).__name__}


@router.get("/stats/monitoring")
async def monitoring() -> dict[str, Any]:
    # Async so the latency windows are read on the event loop, the only place
    # they are written; the blocking queries go to the thread pool.
    latency = _latency_stats()
    rest = await run_in_threadpool(_database_stats)
    return {"generated_at": datetime.now(UTC), "latency": latency, **rest}


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> str:
    return _PAGE


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>edgar-intel monitoring</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --rag: #2a78d6; --agent: #eb6834; --good: #006300; --bad: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --rag: #3987e5; --agent: #d95926; --good: #0ca30c; --bad: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --rag: #3987e5; --agent: #d95926; --good: #0ca30c; --bad: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 0 0 2px; }
.sub, .src { color: var(--ink-2); margin: 0 0 12px; }
.src { font-size: 12px; color: var(--muted); }
section { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; margin-top: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;
  margin-bottom: 12px; }
.tile .v { font-size: 22px; font-weight: 600; }
.tile .k { color: var(--ink-2); font-size: 12px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 5px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th:first-child, td:first-child, td.l, th.l { text-align: left; }
th { color: var(--ink-2); font-weight: 500; font-size: 12px; }
.pending { color: var(--ink-2); font-style: italic; }
.err { color: var(--bad); }
.legend { display: flex; gap: 16px; color: var(--ink-2); font-size: 12px; margin: 4px 0 8px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px;
  vertical-align: -1px; }
#chart { position: relative; }
#chart svg { display: block; width: 100%; height: auto; }
#tip { position: absolute; pointer-events: none; background: var(--surface); color: var(--ink);
  border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; font-size: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,.12); display: none; white-space: nowrap; }
</style>
</head>
<body>
<main>
  <h1>edgar-intel monitoring</h1>
  <p class="sub" id="gen">Loading /stats/monitoring&hellip;</p>

  <section>
    <h2>Latency per endpoint</h2>
    <p class="src" id="lat-src"></p>
    <div class="scroll"><table id="lat"></table></div>
  </section>

  <section>
    <h2>Cost per /ask request</h2>
    <p class="src" id="cost-src"></p>
    <div class="tiles" id="cost-tiles"></div>
    <div class="scroll"><table id="cost-recent"></table></div>
  </section>

  <section>
    <h2>Eval score over time: numeric accuracy</h2>
    <p class="src" id="eval-src"></p>
    <div class="legend"><span><span class="sw" style="background:var(--rag)"></span>RAG (eval run)</span>
      <span><span class="sw" style="background:var(--agent)"></span>agent, as /ask (eval agent)</span></div>
    <div id="chart"><div id="tip"></div></div>
    <div class="scroll"><table id="eval"></table></div>
  </section>

  <section>
    <h2>Data freshness</h2>
    <div class="tiles" id="fresh"></div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const utc = (s) => s ? new Date(s).toISOString().slice(0, 16).replace("T", " ") + " UTC" : "\\u2014";
const ago = (s) => {
  if (!s) return "";
  const h = (Date.now() - new Date(s)) / 36e5;
  return h < 1 ? "under an hour ago" : h < 48 ? Math.round(h) + " h ago" : Math.round(h / 24) + " days ago";
};
const num = (v, d) => v == null ? "\\u2014" : Number(v).toFixed(d);
const usd = (v) => v == null ? "\\u2014" : "$" + Number(v).toFixed(v < 0.01 ? 6 : 2);
const ms = (v) => v == null ? "\\u2014" : Math.round(v).toLocaleString() + " ms";

function table(el, head, rows) {
  el.replaceChildren();
  const tr = el.insertRow();
  head.forEach(([h, l]) => { const th = document.createElement("th"); th.textContent = h;
    if (l) th.className = "l"; tr.appendChild(th); });
  rows.forEach((r) => { const row = el.insertRow();
    r.forEach((c, i) => { const td = row.insertCell(); td.textContent = c;
      if (head[i][1]) td.className = "l"; }); });
}
function tiles(el, items) {
  el.replaceChildren();
  items.forEach(([v, k, cls]) => { const d = document.createElement("div"); d.className = "tile";
    const a = document.createElement("div"); a.className = "v" + (cls ? " " + cls : ""); a.textContent = v;
    const b = document.createElement("div"); b.className = "k"; b.textContent = k;
    d.append(a, b); el.appendChild(d); });
}

const NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  return e;
}
function chart(runs) {
  const box = $("chart"), tip = $("tip");
  box.querySelector("svg")?.remove();
  const W = 960, H = 280, L = 44, R = 100, T = 12, B = 40;
  const pts = runs.filter((r) => r.numeric_accuracy != null);
  const x = (i) => L + (pts.length < 2 ? 0 : i * (W - L - R) / (pts.length - 1));
  const y = (v) => T + (1 - v) * (H - T - B);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Numeric accuracy of each full eval run, in finish order" });
  [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
    svg.appendChild(svgEl("line", { x1: L, x2: W - R, y1: y(v), y2: y(v),
      stroke: v === 0 ? "var(--axis)" : "var(--grid)", "stroke-width": 1 }));
    svg.appendChild(svgEl("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end",
      fill: "var(--muted)", "font-size": 11 }, v.toFixed(2)));
  });
  let lastDay = "", lastX = -1e9;
  pts.forEach((r, i) => {
    const day = r.finished_at.slice(5, 10);
    if (day !== lastDay && x(i) - lastX > 44) {
      svg.appendChild(svgEl("text", { x: x(i), y: H - B + 18, "text-anchor": "middle",
        fill: "var(--muted)", "font-size": 11 }, day));
      lastX = x(i);
      lastDay = day;  // a day squeezed out here gets its label at its next point
    }
  });
  for (const [pipe, color, name] of [["rag", "var(--rag)", "RAG"], ["agent", "var(--agent)", "agent"]]) {
    const idx = pts.map((r, i) => [r, i]).filter(([r]) => r.pipeline === pipe);
    if (!idx.length) continue;
    svg.appendChild(svgEl("polyline", { fill: "none", stroke: color, "stroke-width": 2,
      "stroke-linejoin": "round", points: idx.map(([r, i]) => `${x(i)},${y(r.numeric_accuracy)}`).join(" ") }));
    idx.forEach(([r, i]) => {
      svg.appendChild(svgEl("circle", { cx: x(i), cy: y(r.numeric_accuracy), r: 4, fill: color,
        stroke: "var(--surface)", "stroke-width": 2 }));
      const hit = svgEl("circle", { cx: x(i), cy: y(r.numeric_accuracy), r: 10, fill: "transparent" });
      hit.addEventListener("mouseenter", () => {
        tip.textContent = `${r.label} \\u00b7 ${name} \\u00b7 ${num(r.numeric_accuracy, 4)} \\u00b7 ${utc(r.finished_at)}`;
        tip.style.display = "block";
        const b = box.getBoundingClientRect(), s = box.clientWidth / W;
        tip.style.left = Math.min(x(i) * s + 8, b.width - tip.offsetWidth) + "px";
        const py = y(r.numeric_accuracy) * s;
        tip.style.top = (py < 40 ? py + 12 : py - 34) + "px";
      });
      hit.addEventListener("mouseleave", () => { tip.style.display = "none"; });
      svg.appendChild(hit);
    });
    const [lr, li] = idx[idx.length - 1];
    svg.appendChild(svgEl("text", { x: x(li) + 10, y: y(lr.numeric_accuracy) + 4,
      fill: "var(--ink-2)", "font-size": 12 }, `${name} ${num(lr.numeric_accuracy, 3)}`));
  }
  box.insertBefore(svg, tip);
}

function render(d) {
  $("gen").textContent = "Generated " + utc(d.generated_at) + ". Refreshes every minute. Raw JSON: /stats/monitoring";
  const lat = d.latency;
  $("lat-src").textContent = `Source: ${lat.source}. Window since ${utc(lat.since)}.`;
  table($("lat"), [["Endpoint", 1], ["Requests"], ["OK"], ["p50"], ["p95"], ["4xx"], ["5xx"]],
    Object.entries(lat.endpoints).map(([p, e]) =>
      [p, e.requests, e.ok, ms(e.p50_ms), ms(e.p95_ms), e.client_errors, e.server_errors]));

  if (d.database_error) {
    for (const id of ["cost-src", "eval-src"]) { $(id).textContent = "Database unreachable: " + d.database_error;
      $(id).className = "src err"; }
    return;
  }
  const c = d.cost;
  $("cost-src").textContent = `Source: ${c.source}. ${c.runs} runs, ${utc(c.first_at)} to ${utc(c.last_at)}.`;
  tiles($("cost-tiles"), [
    [usd(c.mean_usd), "mean cost per request"], [usd(c.p95_usd), "p95 cost per request"],
    [usd(c.per_1k_usd), "per 1,000 requests"], [usd(c.total_usd), "total, these runs"],
    [ms(c.p50_latency_ms), "agent p50 latency"], [ms(c.p95_latency_ms), "agent p95 latency"],
  ]);
  table($("cost-recent"), [["Recent run", 1], ["Outcome", 1], ["Latency"], ["Cost"]],
    c.recent.map((r) => [utc(r.created_at), r.outcome, ms(r.latency_ms), usd(r.cost_usd)]));

  const runs = d.eval.runs;
  const last = runs[runs.length - 1];
  $("eval-src").textContent = `Source: ${d.eval.source}. ${runs.length} runs; latest finished ${utc(last?.finished_at)} (${ago(last?.finished_at)}).`;
  chart(runs);
  table($("eval"), [["Finished", 1], ["Run", 1], ["Pipeline", 1], ["Cases"], ["Numeric acc."],
    ["hit@5"], ["MRR"], ["Abstain"], ["Halluc."], ["$ / 1k"], ["git", 1]],
    runs.slice().reverse().map((r) => [utc(r.finished_at), r.label, r.pipeline, r.n_cases,
      num(r.numeric_accuracy, 4), num(r.hit_at_5, 4), num(r.mrr, 4), num(r.abstention_rate, 4),
      num(r.hallucination_rate, 4), num(r.cost_per_1k_usd, 4), r.git_sha || ""]));

  const f = d.freshness, job = f.job;
  const run = job.last_run;
  tiles($("fresh"), [
    job.status === "pending"
      ? ["Pending", "scheduled refresh: not recording runs yet", "pending"]
      : run ? [utc(run.started_at), `last refresh run: ${run.status}, ${run.filings_added ?? 0} filings added (${ago(run.started_at)})`,
               run.status === "ok" ? "" : "err"]
            : ["None yet", "scheduled refresh: no runs recorded"],
    [job.last_ok ? utc(job.last_ok.started_at) : "\\u2014", "last successful refresh"],
    [utc(f.data.newest_ingested_at), `newest filing loaded (${ago(f.data.newest_ingested_at)})`],
    [f.data.newest_filing_date || "\\u2014", "newest filing date at the SEC"],
    [String(f.data.filings), "filings in the database"],
  ]);
}

async function load() {
  try {
    const r = await fetch("/stats/monitoring", { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    render(await r.json());
  } catch (e) {
    $("gen").textContent = "Could not load /stats/monitoring: " + e.message;
    $("gen").className = "sub err";
  }
}
load();
setInterval(load, 60000);
</script>
</body>
</html>
"""
