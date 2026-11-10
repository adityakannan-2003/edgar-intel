# edgar-intel

Retrieval, evaluation, and a bounded agent over SEC EDGAR filings — with
machine-verifiable ground truth.

Most LLM portfolio projects grade themselves with another LLM. That measures
agreement, not correctness, and it drifts silently as models change. This one
is built on a corpus where the SEC publishes the same figures that appear in
the filing text as structured XBRL data. So for any question like *"what was
NVIDIA's FY2024 R&D expense"*, there is an authoritative answer that did not
come from a model — and the evaluation can check it.

That single property is what everything else here rests on.

---

## What's in it

| Component | What it does | What it measured |
|---|---|---|
| **Evaluation harness** | 232-case golden set generated from XBRL, LLM-as-judge for narrative, Cohen's κ calibration, regression gate in CI | Numeric accuracy 0.861. The judge came back at κ = 0.42, below its own 0.60 floor, so its narrative scores are withheld. [Details below](#a-judge-that-has-to-prove-itself) |
| **Hybrid retrieval** | Dense + lexical fused with RRF, an optional cross-encoder reranker, four chunking strategies measured against each other | hit@5 0.762, recall@5 0.408. At an equal 512-token cap, three of the four strategies tied; chunk size mattered more. The reranker added 542 ms at p50 for no gain, so it ships off |
| **Bounded agent** | Six typed tools, step ceiling, retry budget, loop detection, confidence floor, citation + numeric grounding checks | Numeric accuracy 0.41 as first deployed, 0.96 after three single-variable fixes; escalation rate 48% → 2.6% |
| **MCP server** | Same tool registry exposed over MCP, hosted at [`/mcp`](#use-it-from-claude-mcp) | One set of definitions serves the agent, the HTTP API, and any MCP client |
| **Fine-tuning benchmark** | LoRA on structured extraction, benchmarked against few-shot prompting on the same company-disjoint held-out set | The frontier prompt wins on accuracy (0.98 against 0.91 F1); the fine-tune wins only on cost at load. [Results below](#fine-tune-vs-prompting-measured) |
| **Serving + benchmark** | FastAPI, liveness/readiness split, p50/p95/p99, throughput, $/1k requests, concurrency sweep | Deployed `/search`: about 25 req/s on one shared-CPU machine, 0 failures in 1,000 requests |
| **Observability** | OpenTelemetry spans, per-operation token ledger | Every eval run records its exact cost: $0.78 per 1k requests locally |

---

## Use it from Claude (MCP)

The agent's tools are served over MCP at `https://edgar-intel.fly.dev/mcp`.
Nothing to install, no key: every tool is a database read, and none calls a
paid model.

**Claude Code**, one command:

```bash
claude mcp add --transport http edgar-intel https://edgar-intel.fly.dev/mcp
```

**Claude Desktop:** Settings → Connectors → Add custom connector, and paste the
URL above.

| Tool | What it returns |
|---|---|
| `get_fact` | One reported XBRL figure for a company and fiscal year, with its filing accession |
| `compare_fact` | One figure across two fiscal years, with the percentage change |
| `compute_ratio` | One figure divided by another, same company and year |
| `search_filings` | Passages from Risk Factors, MD&A and Business, by hybrid search |
| `list_coverage` | Which companies and fiscal years the database holds |

Try: *"Using edgar-intel, how did Apple's revenue change between fiscal 2024
and 2025, and what does its 10-K say drove it?"*

To run it against your own database instead, `edgar-intel mcp serve` speaks
stdio (`pip install -e '.[mcp]'`).

---

## Quick start

```bash
cp .env.example .env          # set EDGAR_USER_AGENT to a real contact address
make up                       # Postgres + pgvector, OTel collector
make dev                      # install with local models
edgar-intel init              # apply schema

edgar-intel ingest run        # fetch filings + XBRL facts from EDGAR
edgar-intel index build --strategy all
edgar-intel eval build        # generate the golden set from XBRL
edgar-intel eval run --label baseline

edgar-intel agent ask "How did R&D expense change at NVDA between FY2024 and FY2025?"
```

Run everything with **no API key and no spend** by leaving
`EDGAR_LLM_PROVIDER=fake`. The fake provider is not a mock that returns a fixed
string — it answers from the context it is given and grades by real token
overlap, so the pipeline is genuinely exercised end to end. That is what CI
uses.

---

## The part that makes it defensible

### Ground truth that isn't model-generated

`sql/001_schema.sql` → `xbrl_facts`. Every numeric evaluation case is generated
from a row there. Grading is a numeric comparison within a relative tolerance,
with scale tolerance because filings print figures "in millions":

```python
# A model answering "383,285" when the fact is 383,285,000,000 is correct in
# substance. Without scale tolerance this reads as a false failure and gets
# debugged as a model problem for a day.
grade_numeric(case, "383,285")  # -> (True, 1.0, "match at millions scale")
```

### A judge that has to prove itself

The narrative half can't be checked against XBRL, so it uses an LLM judge — and
then measures whether that judge agrees with a human:

```bash
edgar-intel eval label <run_key>   # hand-label answers
edgar-intel eval judge-kappa --contracts v2,v3_1
```

Below κ = 0.60 the report says so explicitly and the gate fails. An
uncalibrated judge is a number generator, and reporting its pass rate as
"quality" is the most common unforced error in LLM evaluation.

**Measured, it failed.** On 24 hand-labelled answers the shipped judge scored
**κ = 0.42** (95% CI 0.12–0.74): 7 false positives, 0 false negatives. It
passed 79% of narrative answers; the human labels pass 50%. So the report marks
narrative scores unusable, and the deployed `/stats/eval` withholds them.
Giving the judge the source passages made it worse (κ 0.58 → 0.42), and three
rubric revisions did not fix it. See
[`docs/METRICS.md` §3](docs/METRICS.md#3-judge-calibration).

### Failures split by cause

```bash
edgar-intel eval failures <run_key>
```

On the reference run:

```json
{
  "total_failures": 32,
  "retrieval_misses": 27,
  "generation_misses": 4,
  "unlabelled_failures": 1,
  "diagnosis": "Most failures are retrieval: the evidence never reached the
                model. Work on chunking, hybrid weighting, or k -- prompt
                changes cannot fix this."
}
```

Conflating these two is how people spend a week tuning prompts to solve a
chunking problem.

### Hallucination caught in code, not by another model

```python
validate_numeric_grounding("Revenue was $999,999 million.", tool_outputs)
# -> (False, "Figures not present in any tool output: ['999,999']")
```

If a figure in the answer never appeared in any tool output, the agent is told
so and made to re-check or escalate. Same for citations: a passage id that no
tool returned is rejected outright.

---

## Fine-tune vs prompting, measured

**The task.** Given one passage from a 10-K, list the financial facts it
states (concept, fiscal year, value) as JSON.
- **Labels:** every label comes from the SEC's XBRL data, not from a model,
  and the scorer is deterministic.
- **Data:** 40 companies in ten sectors, in a database of their own
  (`edgar_ft`).
- **Split:** 1,743 training examples from 30 companies, and 532 held-out
  examples from 10 companies the fine-tune never saw, one per sector.

| arm | micro F1 | $ per 1k requests | p50 / p95 |
|---|---|---|---|
| gpt-6.1-sol, few-shot (2 passes) | **0.980**, 0.980 | $2.07, $1.54 | 2.2 s / 5.5 s |
| gpt-6-luna, few-shot (2 passes) | 0.973, 0.966 | $0.22 | 2.5 s / 11.1 s |
| Qwen3-8B + LoRA (2 training seeds) | **0.908**, 0.912 | $0.58 one at a time, **$0.061 at load** | 1.9 s / 7.1 s |
| Qwen3-8B, few-shot, untuned | 0.439 | $0.68 | 2.1 s / 7.0 s |

**What the numbers say:**
- **Accuracy:** sol beats the fine-tune by 0.07. Its 95% interval is
  [−0.089, −0.056], ten times the largest run-to-run spread measured (0.007,
  luna's two passes). The two LoRA seeds differ by 0.004.
- **What fine-tuning bought:** the same 8B model went from 0.44 to 0.91, at
  0.77 GPU-hours per seed on one RTX 4090. Untuned, it reads the right
  figures and writes "in millions" with one or two zeros too many.
- **Cost:** under load (measured 4.07 requests/s with 32 in flight on a
  $0.89/hr GPU), the fine-tune is about 3.5× cheaper than luna and 25–34×
  cheaper than sol. One request at a time on a dedicated GPU, it costs more
  than luna.
- **Where it breaks:**
  - The fine-tune misses more facts, and reports figures that aren't labels.
  - It invents a few: 16 not in the passage, and facts on 2 of 105 no-fact
    passages, which no API model did.
  - Sol's errors are mostly omissions: 40 of its 70.

> **Resume bullet:** Fine-tuned Qwen3-8B with LoRA on 1,743 XBRL-labelled
> extraction examples (0.77 GPU-hours on one RTX 4090), lifting micro F1 from
> 0.44 (the same model, few-shot) to 0.91 on a 532-case company-disjoint
> held-out set. That is still 0.07 below gpt-6.1-sol few-shot (0.98, beyond
> run-to-run noise of ±0.007), at $0.06 per 1k requests under load against
> $1.54–2.07 for sol and $0.22 for gpt-6-luna.

**How it was run.** The labeller as first written was mostly noise: 18,480 of
its 19,199 labels were diluted EPS. It was rewritten and audited by hand twice:
49 of 50 sampled labels right, a fix, then 50 of 50.

A pilot under a $5 cap ($1.09 spent) found three dataset problems before any
full training run:
1. An instruction that never defined net income or total revenue.
2. Seven companies that tag no `StockholdersEquity` at all.
3. Labels on rows naming a variant of a concept, such as "including
   noncontrolling interests".

Predictions were committed before each paid run and scored afterwards: 7 of
the full run's 10 right, 2 half right, 1 wrong. The full run cost $4.70; the
whole project, $5.87.

Every number above traces to a results file:
- [`docs/METRICS.md` §9](docs/METRICS.md#9-fine-tuning-tradeoff)
- [`docs/FINETUNE_RUN.md`](docs/FINETUNE_RUN.md): results, scored predictions,
  spend
- [`docs/FINETUNE_PILOT.md`](docs/FINETUNE_PILOT.md): what the pilot found and
  changed
- `reports/finetune/full/report.json` (local, gitignored)

---

## Architecture

```
EDGAR API ──► ingest ──► sections ──► chunks (×4 strategies) ──► pgvector
                │                                                    │
                └──► xbrl_facts ──────────┐                          │
                                          │                          ▼
                                          │                  hybrid retrieval
                                          │                   (dense + lexical
                                          │                     RRF → rerank)
                                          │                          │
                                          ▼                          ▼
                                   golden set ◄──────────────► eval harness
                                                                     │
                                    agent ◄── tools ──► MCP          ▼
                                      │                       regression gate
                                      ▼                          (CI blocks)
                                  FastAPI
```

Vectors live in Postgres rather than a dedicated vector database. At this scale
— well under the ~50M-vector mark where purpose-built engines start winning on
index build time and tail latency — pgvector keeps vectors, full text, and the
XBRL ground truth in one transactional store, so a single query can filter by
company and fiscal year *and* rank by vector distance. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the rest of the decisions and
their tradeoffs, [`docs/METRICS.md`](docs/METRICS.md) for every number this repo
has actually measured and an explicit **NOT MEASURED** on the ones it has not,
and [`docs/DEPLOY.md`](docs/DEPLOY.md) for running it somewhere with a URL —
including why the container needs `torch` (queries are embedded at request time,
384-d, so the query model cannot be swapped for an API one) and why `/ask` sits
behind a key and a rate limit while `/search` stays open.

An honest note on the lexical side: it is Postgres full-text search scored with
`ts_rank_cd`, which is tf-idf-family but is **not** BM25 — it has no
document-length saturation term. Calling it BM25 would be wrong. Swapping in
ParadeDB's `pg_search` for true BM25 is a change to one function.

---

## Testing

```bash
make test     # 950 tests, no database, no network, no model downloads
```

Every bound in the agent is tested by making the model misbehave in the
specific way that bound exists to contain — unparseable JSON, infinite tool
loops, invented citations, fabricated figures, low confidence. A limit nobody
has seen fire is a limit nobody knows works.

The CI eval gate (`.github/workflows/ci.yml`) loads a frozen fixture corpus,
builds the index, runs the suite, and fails the build if overall score, numeric
accuracy, or recall@5 regressed by more than 3 points. The gate also blocks any
run whose judge κ is below the floor; CI's run uses a fake model and has no
human labels, so there it checks only those three numbers.

---

## Known gaps

- **The narrative judge is not usable yet** (κ = 0.42). Its errors are all
  false positives, so the likely fix is a missing rule, not a bigger model.
  The 9-repeat calibration run is committed but not yet run.
- **Most remaining failures are retrieval.** Of the reference run's 29 numeric
  failures, 24 never had the expected figure in the model's context.
- **The corpus is small:** 8 companies, 24 10-K filings, FY2023–FY2026.
- **Untried:** a retriever that fuses both embedders, and a paid
  single-variable reranker run on the current label set.

---

## Cost

Retrieval, chunking, indexing and all retrieval metrics run **free** — local
embeddings on CPU, lexical search in Postgres. Only generation and judging cost
money. A full 232-case evaluation run against a small frontier model is
typically well under a dollar; the harness reports exact `total_cost_usd` and
`cost_per_1k_usd` on every run, so you never have to guess.

The fine-tune benchmark cost **$5.87** in total: $2.73 of API calls and
$3.14 of rented GPU time, from RunPod's billing read. The full run was $4.70
of that, and the pilot $1.09.

## Licence

MIT. The SEC's EDGAR data is public domain; their
[access policy](https://www.sec.gov/os/webmaster-faq#developers) requires a
descriptive User-Agent with a contact address and a rate limit under 10
requests/second, both enforced in `ingest/edgar_client.py`.
