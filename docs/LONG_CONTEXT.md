# RAG against long context: plan and predictions

Written 30 Oct 2026, before the experiment's first paid call. The
predictions below are scored against the results files afterwards, right or
wrong. The machine-readable copy is
`reports/long_context/predictions-20261030.json`.

## The question

Does the pipeline need retrieval at all, now that a model's context holds a
whole 10-K? The same model answers each golden-set question two ways:

| arm | what the model reads |
|---|---|
| `rag` | the shipped retrieval: hybrid pgvector + full-text, RRF (k=60), `section_aware` chunks, top 8 under the 20,000-character cap. The same `search()` and `build_context_report()` calls `eval run` makes. |
| `long_context` | every section of the 10-K the question is about (the company and fiscal year the RAG arm filters on), in filing order, tagged the way retrieved passages are |

Everything else is held fixed: `ANSWER_SYSTEM` and `ANSWER_TEMPLATE` from
`eval run`, the structured-output schema, the model, its reasoning effort, the
grader.

## What is fixed

- **Set:** `evalset/golden.json`, 232 cases: 160 numeric single-hop, 48
  numeric comparative, 24 narrative. 24 filings, 6 to 14 questions each.
- **Model: gpt-6-luna**, reasoning effort `medium` (its default), Responses
  API, in both arms. The pipeline's own gpt-4o-mini cannot run the long
  arm: its window is 128K tokens, and the three CAT 10-Ks are about 140K
  (565K characters). luna's window is 1.05M. Of the models the repo already
  prices, luna is the same tier gpt-4o-mini fills, and it is cheap enough to
  run each arm twice, which measures the noise floor. GPT-6 models accept no
  temperature but the default 1, so two passes are needed to tell a
  difference from noise.
- **Request shape**, both arms: system `ANSWER_SYSTEM`; user message 1, the
  context (`CONTEXT:` up to the question); user message 2, `QUESTION:` and the
  reply instructions. The two user messages join to exactly the prompt `eval
  run` sends (`tests/test_long_context.py` holds the RAG arm's prompt equal to
  `answer_question`'s).
- **Caching.** On GPT-6 models, implicit caching writes at the end of the whole
  request, question included, so a shared document is reusable only behind an
  explicit breakpoint (developers.openai.com/api/docs/guides/prompt-caching,
  read 30 Oct). The long arm sends `prompt_cache_options: {"mode":
  "explicit"}` with a `prompt_cache_breakpoint` after the filing and
  `prompt_cache_key` set to the filing's accession. The RAG arm sends explicit
  mode with no breakpoint, which writes nothing: its contexts are never
  reused, and it should not pay a write surcharge for them.
- **Order: questions run grouped by filing** (ticker, then fiscal year, set
  order within), one at a time, so each filing is written to the cache once
  and read by the rest of its questions within minutes. A cold pass can at
  best read the cache on 208 of 232 requests. The order is the same in both
  arms.
- **Prices** (developers.openai.com/api/docs/pricing, read 30 Oct; every
  request here is under the 272K-token tier): luna $0.10 input, $0.01 cache
  read, $0.125 cache write (30-minute TTL), $0.50 output per 1M tokens. Cost
  is list price times the tokens each response reports.
- **Grading:** numeric cases by `grade_numeric`, the XBRL grader `eval run`
  uses, 0.5% tolerance; abstention by `is_abstention`. Narrative answers are
  kept and not judged: the judge's pass rate is not usable (κ 0.42, METRICS
  §3), and the configured judge, gpt-4o-mini, cannot hold the long arm's
  source either.
- **Results** go to `reports/long_context/*.jsonl` (one record per case, with
  usage, latency, the retrieved context for RAG, and the grade), never to
  `eval_runs`, so `/stats/eval` and `eval gate` cannot pick them up.

## Runs

1. **Pilot** (about $2): AAPL FY2024, CAT FY2025, JNJ FY2024 (38 cases, 35
   numeric) in both arms on luna, covering the smallest and the two largest
   filings. Also AAPL FY2024 and CAT FY2025 (24 cases) on gpt-6.1-sol, so the
   choice of model for the full run rests on measured cost and cache
   behaviour, not on a projection. Then stop and report.
2. **Full run**, only after Aditya's go: on luna, long context pass 1, RAG
   pass 1, RAG pass 2, long context pass 2. Each pass takes about half an
   hour, so a filing's entry from pass 1 has expired (30-minute TTL) before
   pass 2 reaches it, and both long-context passes are cold.

```bash
# from the main checkout (pydantic reads .env from the cwd), this branch's code:
PYTHONPATH=<worktree>/src edgar-intel eval long-context run --arm long_context \
  --out reports/long_context/luna-lc-p1.jsonl
PYTHONPATH=<worktree>/src edgar-intel eval long-context report \
  reports/long_context/luna-rag-p1.jsonl reports/long_context/luna-lc-p1.jsonl ... \
  --out reports/long_context/report.json
```

## How results are read

- **Accuracy** is numeric accuracy over the 208 numeric cases, with a 95%
  Wilson interval per pass.
- **The arm difference** is long context pass 1 against RAG pass 1, with a
  paired bootstrap 95% interval over cases, confirmed on pass 2 against pass 2.
- **A difference is claimed** only when its paired interval excludes zero and
  it is larger than the larger same-arm difference (RAG pass 2 − pass 1, long
  context pass 2 − pass 1).
- **Cost per 1,000 questions** is from the cold pass, at this set's 9.7
  questions per filing. Because caching makes the long arm's cost depend on
  how many questions share a filing, the write-up also gives it at one
  question per filing and with an unlimited number of them, computed from
  the measured tokens.
- **Latency:** p50 and p95, end to end: retrieval plus the model call for RAG,
  the model call for long context (each filing is read from the database once
  per pass and that read is counted on its first question).
- **Cache hit rate:** the share of long-context requests that read the cache,
  and the share of input tokens read from it.
- **Failures** are read one by one and grouped by cause, as
  `docs/FINETUNE_RUN.md` does. For RAG, the first split is whether the
  expected figure was in the retrieved context at all (the method in METRICS
  §5).
- **Failed calls** are retried by resuming and never scored as wrong answers.
  If more than 2% of an arm's cases still fail, the arm is reported as
  incomplete.

## Stop rules

- The pilot stops after its runs, and the full run waits for Aditya's go.
- If the full run's spend passes $5 on luna, it stops and is reported as it
  stands.

## Predictions

Compared with the shipped pipeline, `exp-prompt-conventions-c0f6e105` on
gpt-4o-mini: 0.8606 (179/208), abstention 0.0865, wrong 0.0529, $0.78 per 1k,
p50 1.2 s / p95 2.9 s. 24 of its 29 numeric failures never had the figure in
its context.

**Pilot**
1. luna costs $0.05–0.15 for the long arm and $0.02–0.07 for RAG; sol $0.6–1.3
   and $0.3–0.8. Total pilot spend $1.0–2.4.
2. The long arm writes the cache 3 times on luna (once per filing) and reads it
   on at least 33 of the 35 later requests; on sol, at least 20 of 22. RAG
   writes nothing.

**Full run on luna, two passes per arm**
3. RAG numeric accuracy is 0.84–0.90, and long context 0.87–0.95.
4. Long context beats RAG by 0.00 to +0.08, and the difference is claimed under
   the rule (about 60% confident).
5. Long context passes 10–25 cases RAG fails, and RAG passes 3–12 that long
   context fails.
6. Each arm's two passes differ by at most 0.03 and flip 4–20 cases in total.
7. Comparative: RAG 40–46 of 48, long context 42–47.
8. RAG abstains on 4–10% of numeric cases and answers wrongly on 3–7%. Long
   context abstains on 0–3% and answers wrongly on 4–10%: with the figure
   always present, it rarely says it is missing, and its errors move from
   abstaining to picking the wrong one.
9. 60–90% of RAG's numeric failures never had the expected figure in the
   retrieved context.
10. Long context's failures are mostly a wrong line, column or year in a table
    where the right figure is also printed (the prior-year column, a segment
    row, the noncontrolling-interest line), not missing evidence, and 35–65%
    of them are on CAT or JNJ, the two largest filings (25% of the questions).
11. Cost per 1k questions: long context $1.80–3.20, RAG $0.60–1.30. Without the
    cache, long context would cost $8–11 per 1k.
12. The cold long-context pass reads the cache on 85–90% of requests and 85–90%
    of input tokens (the ceiling is 208/232 = 89.7% of requests).
13. RAG p50 2.5–6 s, p95 6–15 s. Long context p50 4–12 s, p95 10–30 s; its p50
    is 1.3–3× RAG's. Its mean output tokens are 1–2× RAG's.
14. After resuming, at most 2 calls stay failed or incomplete.
15. Total spend $1.0–2.5.

If sol is chosen instead (one pass per arm): long context $8–10, RAG $3–5;
long context 0.90–0.97, RAG 0.85–0.91.
