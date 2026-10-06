# Deploying edgar-intel

The service is a FastAPI app (`edgar_intel.serving.app`) over a Postgres
database that must have **pgvector**. Everything hard about deploying it comes
from two facts:

- **Queries are embedded at request time**, locally, with
  `all-MiniLM-L6-v2` (384-d). The indexed vectors are 384-d, so the query
  embedder cannot be swapped for an API embedding model — OpenAI's smallest is
  1536-d and the cosine distances would be meaningless. The container therefore
  needs `torch` + `sentence-transformers`, and about 1 GB of RAM.
- **The database is not empty on first boot.** It needs the filings ingested,
  chunked and embedded. That is a one-time load run from a machine that can
  reach `sec.gov`.

## What gets exposed

| endpoint | cost | public |
|---|---|---|
| `GET /health` | free | yes — liveness, no database dependency |
| `GET /ready` | free | yes — database + the served strategy's index + embedder |
| `GET /config` | free | yes — non-secret runtime config |
| `GET /stats/eval` | free | yes — the latest evaluation summary; the judge's fields withheld unless κ clears the floor |
| `GET /stats/agent` | free | yes |
| `POST /search` | free | yes — local embeddings and Postgres only |
| `POST /ask` | **paid** | **key + rate limit** — runs the agent against a paid model |
| `GET /traces/{id}` | free | yes |

`/stats/eval` is the reason a live URL is worth having on a resume: the numbers
in `docs/METRICS.md` are served by the same instance that answers the questions,
so "how good is this right now" is one request away.

**`/ask` spends this deployment's own API key.** Set `EDGAR_API_KEY` and keep
`EDGAR_API_RATE_LIMIT_PER_MIN` non-zero before the URL is public. Both controls
are inert when unset so local development and `docker-compose` are unchanged —
which means forgetting them is silent. Check `GET /config`: it reports
`ask_requires_api_key` and `ask_rate_limit_per_min`.

## 1. Database — managed Postgres with pgvector

Fly's managed Postgres does not ship pgvector, and `sql/001_schema.sql` opens
with `CREATE EXTENSION vector`. Use a provider that has it. Neon and Supabase
both do on their free tiers.

```bash
# Neon: create a project, copy the pooled connection string.
export REMOTE_DSN='postgresql://USER:PASS@HOST/DB?sslmode=require'
psql "$REMOTE_DSN" -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;"
psql "$REMOTE_DSN" -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','pg_trgm');"
```

Both rows must come back — `CREATE EXTENSION` fails silently under a role
without permission on some hosts. Note pgvector's version; §2d may need it.

**Do not run `edgar-intel init` against it.** The restore in §2 creates the
schema, and a database whose tables already exist makes that restore stop on
"relation already exists".

## 2. Load the corpus — copy it, do not rebuild it

The local database already holds everything the deployed instance needs: 24
filings, 264 sections, 1,338 XBRL facts, the `section_aware` index the service
reads — 5,468 chunks, every one embedded — and the eval-run history that
`/stats/eval` serves. **Copy that, and only that.**

Re-running `ingest` and `index build` against a remote DSN also works and is the
wrong choice. It re-fetches from `sec.gov`, re-embeds every chunk on CPU, and
then writes every 384-dimensional vector over a home uplink one statement at a
time — an hour or more, for a database that is byte-for-byte reproducible from
the one already on disk. Worse, a fresh ingest can pick up *newer* filings than
the ones the current metrics were measured against, so `/stats/eval` would report
numbers from a corpus the instance is no longer serving.

### 2a. Before the copy

- **Exercise `/ask` locally.** The agent has been measured on the full golden
  set (`eval agent`, `docs/METRICS.md` §6), but those runs do not write
  `agent_traces`, which still holds only 2 manual runs from 28 Sep. A few
  `edgar-intel agent ask "…"` calls against the local database cost cents and
  leave traces that the copy then carries to `/stats/agent`. Deploy at or after
  `c4a7424`: before it, a citation-format bug escalated 48% of questions.
- **Record no evaluation run after the one you mean to publish.** `/stats/eval`
  serves the most recently *finished* retrieve-then-answer run — today
  `exp-prompt-conventions-c0f6e105`, the reference run in `docs/METRICS.md`.
  Agent runs carry `pipeline: agent` and are skipped.

### 2b. Make a trimmed copy

The local `chunks` table also holds three strategies the service never reads
(`fixed`, `recursive`, `semantic`) and the `section_aware_t144` experiment —
about 50,000 rows the deployed database would store and never query. Trim a copy
rather than the original, so the local database keeps every experiment:

```bash
export LOCAL=postgresql://edgar:edgar@localhost:5433
# TEMPLATE needs no other session on `edgar`: stop the API, close any psql first
psql "$LOCAL/postgres" -c "DROP DATABASE IF EXISTS edgar_deploy;"       # the copy, never `edgar`
psql "$LOCAL/postgres" -c "CREATE DATABASE edgar_deploy TEMPLATE edgar;"
psql "$LOCAL/edgar_deploy" -c "DELETE FROM chunks WHERE strategy <> 'section_aware';"
psql "$LOCAL/edgar_deploy" -c "SELECT strategy, count(*) FROM chunks GROUP BY 1;"   # section_aware | 5468
psql "$LOCAL/edgar_deploy" -c "SELECT pg_size_pretty(pg_database_size(current_database()));"
```

### 2c. Copy it

```bash
# 1. dump the trimmed copy -- schema, data, and the HNSW index definition
pg_dump "$LOCAL/edgar_deploy" --no-owner --no-privileges --no-comments -f /tmp/edgar.sql

# 2. restore into the empty database
psql "$REMOTE_DSN" -v ON_ERROR_STOP=1 -f /tmp/edgar.sql
```

If the restore fails on `CREATE EXTENSION` because the role lacks permission, the
extensions from §1 are already there — re-run with those lines stripped:
`grep -v 'CREATE EXTENSION' /tmp/edgar.sql > /tmp/edgar-noext.sql`.

Two things to check before trusting it:

```bash
# counts: 24 filings / 264 sections / 5468 chunks / 5468 embedded / 1338 facts
psql "$REMOTE_DSN" -c "SELECT 'filings' t, count(*) FROM filings
  UNION ALL SELECT 'sections', count(*) FROM sections
  UNION ALL SELECT 'chunks', count(*) FROM chunks
  UNION ALL SELECT 'embedded', count(*) FROM chunks WHERE embedding IS NOT NULL
  UNION ALL SELECT 'xbrl_facts', count(*) FROM xbrl_facts;"

# the HNSW index has to exist on the target, and a dump may not carry it usefully
psql "$REMOTE_DSN" -c "\di+ chunks_embedding_idx"
```

If the HNSW index is missing, rebuild it **after** the rows are loaded — building
it on an empty table and then inserting is the slow order:

```bash
psql "$REMOTE_DSN" -c "CREATE INDEX IF NOT EXISTS chunks_embedding_idx
  ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);"
```

The `m` and `ef_construction` values are not decoration — they must match
`sql/001_schema.sql`. Building the deployed index with pgvector's defaults would
give the live instance different recall characteristics from the one every number
in `docs/METRICS.md` was measured on, and nothing would report the difference.

`embedded` must equal `chunks`. A deployed instance with an empty or partial
`section_aware` index is what `/ready` catches — it returns 503 with
`index: false` rather than serving empty results.

### 2d. Check retrieval on the target, not just the row counts

Row counts miss the one failure specific to this copy. Dense retrieval filters on
strategy, ticker and fiscal year and orders by vector distance. If Postgres
answers that from the HNSW index, pgvector applies the filters *after* the index
returns its `hnsw.ef_search` candidates — 40 by default — so a query filtered to
one company-year can get a handful of rows instead of k, with no error. Locally
the planner has not done this (dense-only reaches 0.810 in the 26 Sep sweep), but
the deployed database is smaller and differently shaped. Measure it:

```bash
EDGAR_DB_DSN="$REMOTE_DSN" edgar-intel eval retrieval --out reports/retrieval_neon.json
```

It embeds the queries locally and only reads the remote database. Compare it
with `reports/retrieval_current_d7.json` configuration by configuration — match
rows on `mode`, `rerank`, `k` and `top_n`, because the newer sweep names its
rows from the `use_rerank` setting. Same data, same queries: the figures should
agree. Ties in the lexical ranking can move a case or two; a drop in the
dense-only row is the signal this check exists for.

If dense-only falls, turn on pgvector's iterative index scans for the database
(pgvector 0.8 or later), so a filtered scan keeps going until it has k rows, and
measure again:

```bash
psql "$REMOTE_DSN" -c "ALTER DATABASE <dbname> SET hnsw.iterative_scan = strict_order;"
```

On an older pgvector, stop there: the query has to change, not the settings.

**Storage:** trimmed, `chunks` is 5,468 rows — about 10 MB of text and 8 MB of
vectors — plus the `body_tsv` GIN and HNSW indexes, the sections and the eval
history. The size query in §2b gives the real figure; check it against the
provider's free-tier limit before the restore rather than halfway through it.

## 3. Deploy

```bash
fly launch --no-deploy                  # accept the existing fly.toml
fly secrets set \
  EDGAR_DB_DSN='postgresql://...' \
  EDGAR_LLM_API_KEY='sk-...' \
  EDGAR_API_KEY="$(openssl rand -hex 24)"
fly deploy
```

The image build installs torch from the CPU wheel index and pre-downloads both
models, so it is large and slow to build once, then fast to start.

Leave `EDGAR_USE_RERANK` unset. Its default, `false`, is the configuration every
number in `docs/METRICS.md` describes; on a shared CPU the cross-encoder would
also cost well over the 540 ms it cost on the Mac. The reranker model is still in
the image, so a request can ask for `"rerank": true` explicitly.

## 4. Verify — in this order

```bash
curl -fsS https://edgar-intel.fly.dev/health
curl -fsS https://edgar-intel.fly.dev/ready          # ready: true, strategy section_aware, indexed_chunks 5468
curl -fsS https://edgar-intel.fly.dev/config         # ask_requires_api_key: true, use_rerank: false
curl -fsS https://edgar-intel.fly.dev/stats/eval     # exp-prompt-conventions; judge_calibration.withheld
                                                     #   lists narrative_pass_rate and overall_score

curl -fsS -X POST https://edgar-intel.fly.dev/search \
  -H 'content-type: application/json' \
  -d '{"query":"supplier concentration risk","ticker":"AAPL","top_n":5}'
                                                     # "reranked": false

# Should be 401 without the key:
curl -si -X POST https://edgar-intel.fly.dev/ask \
  -H 'content-type: application/json' -d '{"question":"What was Apple FY2025 revenue?"}' | head -1

# And work with it:
curl -fsS -X POST https://edgar-intel.fly.dev/ask \
  -H "x-api-key: $EDGAR_API_KEY" -H 'content-type: application/json' \
  -d '{"question":"What was Apple FY2025 revenue?"}'
```

`/ready` returning `ready: true` while `/search` fails is the exact failure this
project already hit once: the first Dockerfile installed `.[serving,obs]` without
`.[models]`, so the embedder raised `ImportError` on every query while both
probes reported healthy. **If `/ready` says `embedder: false`, read
`embedder_error` — it distinguishes a missing package from a dimension mismatch,
and a dimension mismatch means the index and the query model disagree, which no
amount of restarting will fix.**

## Cost

| | |
|---|---|
| Fly machine, 1 GB shared-cpu-1, suspending when idle | free allowance covers a portfolio deployment |
| Neon / Supabase free tier | 0 |
| `/search` | 0 — local embeddings, Postgres only |
| `/ask` | not measured: `/ask` runs the agent, which takes several model calls per question. The single-call evaluation path costs $0.78 per 1k questions at `gpt-4o-mini`, so expect a multiple of that. |

The rate limit at 20/min per IP caps a single abusive client at roughly $0.60 an
hour. That is the number the limit was chosen against, and it is why the limit is
per-IP rather than global: a global limit would let one client deny the endpoint
to everyone else for the price of a rounding error.

## Known limitations, stated rather than hidden

- The rate limiter is **in-process**, so it is per-machine: two instances allow
  twice the budget. Correct at `min_machines_running = 0` with one machine; a
  shared counter would mean Redis.
- Cold start pays the model load. `/ready` embeds a probe string to warm it, but
  the first request after a suspend is still slower than p95 from the evaluation
  runs, which were measured against a warm local process.
- `p50 892 ms / p95 1732 ms` in `docs/METRICS.md` are **local** figures. Do not
  quote them for the deployed instance until `edgar-intel bench sweep` has run
  against the URL.
