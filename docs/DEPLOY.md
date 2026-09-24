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

## 2. Load the corpus

Run from a machine that can reach `sec.gov` — in this project's environment that
means the local Mac, since `sec.gov` is blocked by egress policy from cloud
containers. Pointing `EDGAR_DB_DSN` at the remote database and running the
ingest locally is the whole trick: the slow, network-bound work happens where
the network works, and the deployed instance only reads.

```bash
edgar-intel ingest run                      # ~24 filings, 8 companies, 3 years
edgar-intel ingest verify-facts             # must come back clean
edgar-intel index build --strategy all      # chunks + embeddings, the slow step
edgar-intel status                          # record these counts
```

`index build` writes every embedding over the wire to the remote database; on a
home connection expect this to be the longest step by far. It is also the step
whose absence `/ready` catches: a deployed instance with an empty `chunks` table
returns 503 with `index: false` rather than serving empty results.

Optional, and worth it — `/stats/eval` has nothing to report until an evaluation
run exists in the remote database:

```bash
edgar-intel eval build
edgar-intel eval run --label deployed
```

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
