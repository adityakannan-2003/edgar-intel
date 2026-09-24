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
| `GET /ready` | free | yes — database + index + embedder |
| `GET /config` | free | yes — non-secret runtime config |
| `GET /stats/eval` | free | yes — the latest evaluation summary |
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
export EDGAR_DB_DSN='postgresql://USER:PASS@HOST/DB?sslmode=require'
edgar-intel init                   # creates the extensions and the schema
```

Check the extension actually landed — `CREATE EXTENSION` fails silently under a
role without permission on some hosts:

```bash
psql "$EDGAR_DB_DSN" -c "SELECT extname FROM pg_extension WHERE extname IN ('vector','pg_trgm');"
```

Both rows must come back. `pg_trgm` is also required.

## 2. Load the corpus — copy it, do not rebuild it

The local database already holds everything the deployed instance needs: 24
filings, 264 sections, 23,499 chunks across four strategies, every one embedded,
1,338 XBRL facts and the eval-run history that `/stats/eval` serves. **Copy that.**

Re-running `ingest` and `index build` against a remote DSN also works and is the
wrong choice. It re-fetches from `sec.gov`, re-embeds all 23,499 chunks on CPU,
and then writes every 384-dimensional vector over a home uplink one statement at
a time — an hour or more, for a database that is byte-for-byte reproducible from
the one already on disk. Worse, a fresh ingest can pick up *newer* filings than
the ones the current metrics were measured against, so `/stats/eval` would report
numbers from a corpus the instance is no longer serving.

```bash
# 1. extensions first -- a restore cannot create a vector column without them
psql "$REMOTE_DSN" -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;"

# 2. dump local, excluding the extension statements the target already has
pg_dump "postgresql://edgar:edgar@localhost:5433/edgar" \
  --no-owner --no-privileges --no-comments \
  -f /tmp/edgar.sql

# 3. restore
psql "$REMOTE_DSN" -v ON_ERROR_STOP=1 -f /tmp/edgar.sql
```

If step 3 fails on `CREATE EXTENSION` because the role lacks permission, the
extensions from step 1 are already there — re-run with those lines stripped:
`grep -v 'CREATE EXTENSION' /tmp/edgar.sql > /tmp/edgar-noext.sql`.

Two things to check before trusting it:

```bash
# counts must match `edgar-intel status` locally: 24 / 264 / 23499 / 1338
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
`chunks` table is what `/ready` catches — it returns 503 with `index: false`
rather than serving empty results.

**Storage:** the corpus is roughly 45 MB of chunk bodies, ~36 MB of vectors, plus
the `body_tsv` GIN index and the HNSW index. That fits a Neon or Supabase free
tier, but check the project's limit before the restore rather than halfway through
it.

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

## 4. Verify — in this order

```bash
curl -fsS https://edgar-intel.fly.dev/health
curl -fsS https://edgar-intel.fly.dev/ready          # expect ready: true
curl -fsS https://edgar-intel.fly.dev/config         # ask_requires_api_key: true
curl -fsS https://edgar-intel.fly.dev/stats/eval

curl -fsS -X POST https://edgar-intel.fly.dev/search \
  -H 'content-type: application/json' \
  -d '{"query":"supplier concentration risk","ticker":"AAPL","top_n":5}'

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
| `/ask` | ~$0.0005 per request at `gpt-4o-mini` (measured: $0.5435 per 1k) |

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
- `p50 996 ms / p95 1700 ms` in `docs/METRICS.md` are **local** figures. Do not
  quote them for the deployed instance until `edgar-intel bench sweep` has run
  against the URL.
