# Monitoring

Pipeline stage 9 is "monitoring and cost": tracing, a dashboard, and cost per
request on the live system. The tracing was already there (OpenTelemetry to
Grafana Cloud). This adds the dashboard.

- **Page:** https://edgar-intel.fly.dev/dashboard (reloads its data every minute)
- **Data behind it:** https://edgar-intel.fly.dev/stats/monitoring (read-only JSON)
- **Code:** `src/edgar_intel/serving/monitoring.py`; tests in `tests/test_monitoring.py`
- **Snapshot:** `docs/monitoring/` holds an export taken on 11 Nov 2026, before
  this was deployed: the JSON and a screenshot of the page drawn from it. The
  local app read the live Neon database with every transaction forced
  read-only. The latency panel in it is empty because the latency window
  belongs to the process serving the page, and that was a local one.

## Why a page served by the app, not a Grafana dashboard

The obvious alternative was a Grafana Cloud dashboard over the traces, with
TraceQL metrics for the percentiles and the Infinity plugin reading a stats
endpoint for the rest. It was rejected for three reasons.

1. **Most of the data is not in the traces.** Cost per request, eval history and
   the refresh record live in Postgres. A Grafana dashboard would still need a
   new endpoint, plus a plugin to read it, so it adds parts and removes none.
2. **TraceQL metrics suit busy services, and this one is quiet.** A TraceQL
   metrics query is limited to a short window (3 hours by default in Tempo),
   and the live service gets a few requests a week. A p95 over that window
   would be blank nearly every time anyone looked.
3. **It can be tested and checked without anyone opening Grafana.** Every
   number on the page comes from one endpoint, which has tests in CI. Anyone
   can check it with `curl`, with no plugin to install, no data source to set
   up and no Grafana credentials.

The cost: when the app is down, so is its dashboard. Grafana still has the
traces, and Fly's health checks still fire, so an outage stays visible. The
latency window has its own limit, described under its panel.

## The panels

### 1. Latency per endpoint: p50 and p95 for /ask, /search, /mcp

| | |
|---|---|
| Source | A middleware (`record_latency`) times every `POST` to the three paths and keeps the last 1,000 per endpoint in memory |
| Measured | Request start to response ready. All three return complete JSON (`/mcp` is stateless with JSON responses), so this is the full server-side time |
| Excluded from p50/p95 | 4xx and 5xx responses, counted in their own columns. A burst of fast 401s from callers without a key would otherwise drag `/ask`'s p50 down |
| `/mcp` | Every MCP HTTP request counts, including `initialize` and `tools/list`, not only `tools/call` |
| Percentile | Nearest rank, the same function as `bench` (`serving/bench.py`). An empty window shows "—", never 0 ms |

**Limit, stated in the code as a shortcut:** the window is per process. A
restart or deploy empties it, and every push to `main` deploys, docs-only pushes
included. A second machine would keep its own window. If history across deploys
is ever needed, the upgrade is a `request_log` table. It was not built because
it means a write on every request to the production database. For longer
history today, use the traces in Grafana (below).

### 2. Cost per /ask request

| | |
|---|---|
| Source | `agent_traces`, the newest 500 rows. Every agent run records its total cost and latency there; it also serves `/stats/agent` and `/traces/{id}` |
| Tiles | mean, p50, p95 cost per request; cost per 1,000 requests (mean × 1,000); total; agent p50/p95 latency |
| Table | the 10 newest runs: time, outcome, latency, cost |
| Not shown, on purpose | question text and trace ids. `/traces/{id}` returns a visitor's question, so listing ids would make every question public |

`/search` and `/mcp` call no paid model (local embeddings, Postgres reads), so
their model cost is zero and they get no cost panel. The Fly machine
(about $6.70 a month, `docs/DEPLOY.md`) is a fixed cost and not included.

On 11 Nov: **9 runs** since 28 Sep, **mean $0.000453 per request**
($0.45 per 1,000), p95 $0.001101, total $0.004073. The agent's p50 latency was
5.6 s and its p95 17.1 s.

### 3. Eval score over time

| | |
|---|---|
| Source | `eval_runs` in the live database: finished runs with at least 100 cases, ordered by finish time |
| Chart | numeric accuracy per run, one line for RAG (`eval run`) and one for the agent (`eval agent`, what `/ask` serves). The x-axis is run order, labelled with dates, because 12 of the 25 runs finished on 6 Oct |
| Table | every run: numeric accuracy, hit@5, MRR, abstention, hallucination, cost per 1k, git sha |
| Excluded | smoke, pilot and retry runs (under 100 cases). They are real runs, but they would look like swings in quality |
| Never selected | `narrative_pass_rate` and `overall_score`. Both come from the LLM judge, which `/stats/eval` withholds until its kappa clears the floor. This query never reads them. A test checks the SQL |

How to read it:

- The three near-zero runs on 12 Sep are void, not low quality. METRICS §2
  lists them: `baseline-45536dc9` got a 401 on every case, `baseline-f6da95f6`
  is void, and `baseline-c8d68dd6` ran on the fake provider. The chart keeps
  them because the database records them; the table names each one.
- RAG and agent figures are different pipelines. `docs/METRICS.md` explains why
  the agent's 0.9615 does not mean better reading than RAG's 0.8606.
- Each run's figures are as graded when it finished. Re-grades
  (`reports/regrade-*.json`, METRICS §6) are not written back. That is why the
  agent's abstention and hallucination rates here differ from §6's re-graded ones.
- The history is only as fresh as the runs in the live database, and it ends on
  7 Oct. Evals run on a laptop and were copied to Neon at deploy. The panel
  shows "latest finished … (N days ago)", so you can see how old it is.
- `reports/*.json` holds local exports of these same runs. It is gitignored
  (only `reports/serving/` is committed), and the image does not include it, so
  the database is the source.

### 4. Data freshness and last refresh

| | |
|---|---|
| Data | `filings`: count, newest `created_at` (when a filing was last loaded) and newest `filing_date` (its SEC date) |
| Refresh job | the newest row in `refresh_runs`, and the newest with `status = 'ok'` |

The scheduled SEC refresh (`docs/REFRESH.md`, merged in PR #6) writes one row per run to `refresh_runs`
in the live database, created by its `sql/006_refresh_runs.sql`. This panel
reads `started_at`, `finished_at`, `status` and `filings_added`:

| status | meaning |
|---|---|
| `ok` | finished cleanly: nothing new, or new filings validated, gated and published |
| `blocked` | bad data or a failed eval gate stopped the publish; live data unchanged |
| `failed` | the job crashed (network, SEC error); live data unchanged |

**Pending until that branch merges and the job runs.** Until the table exists,
the panel says "Pending". The endpoint checks with `to_regclass`, so no
code change is needed when the table arrives. Both columns matter. A run that
finds nothing new changes no filing, so `filings.created_at` alone would look
stale even when the job is healthy.

On 11 Nov: 24 filings, newest loaded 11 Sep, newest SEC filing date 4 Aug 2026.

## When the database is down

`/stats/monitoring` still answers 200 with the latency panel. The other panels
are replaced by `database_error`, which gives only the exception's type, since a
driver's message can contain the database host.

## Grafana: for one request, not the overview

Traces still go to Grafana Cloud (`obs/tracing.py`). Spans: `api.ask`,
`api.search` and `mcp.call_tool` (one per tool call, with the tool name). To
look at slow requests, open **Explore**, pick the Tempo (traces) data source
and run:

```
{resource.service.name="edgar-intel" && name="api.search"}
```

Swap in `api.ask` or `mcp.call_tool`. Nothing has to be set up in Grafana for
the dashboard.

## Check it

```bash
curl -s https://edgar-intel.fly.dev/stats/monitoring | python3 -m json.tool | head -40
```

```bash
PYTHONPATH=src python -m pytest tests/test_monitoring.py
```
