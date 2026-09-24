# From measurement to resume bullet

Every number this repo produces, where it comes from, and the bullet it
supports — with the figures actually measured, and an explicit **NOT MEASURED**
on the ones that have not been.

> **The rule that governs this whole document:** run the command, record what it
> actually says, and write the bullet from that. A placeholder left in a resume
> is a claim you cannot defend, and the interviewer will pick exactly the most
> impressive-sounding line to probe first.

## Provenance of every number below

| | |
|---|---|
| Reference run | `baseline-v5-2943b37a`, 17 Sep 2026, `git_sha` `a9e9c08` |
| Report | `reports/baseline-v5-2943b37a.json` |
| Set | 232 cases — 160 numeric single-hop · 48 numeric comparative · 24 narrative |
| Corpus | 8 companies (AAPL CAT COST JNJ MSFT NVDA PG UNH), **24 10-K filings**, FY2023–FY2026 · 264 sections · 1,338 XBRL facts · 23,499 chunks across four strategies, all embedded (`section_aware` 5,468) — confirmed with `edgar-intel status`, 24 Sep |
| Config | `section_aware` · `hybrid` · `use_rerank=false` · `k=50` · `top_n=8` · `gpt-4o-mini` · MiniLM-L6-v2 (384-d) · `rrf_k=60` · `numeric_tolerance=0.005` · `context_max_chars=12000` |

Two standing cautions, both earned the hard way:

- **Read the `config` block before the numbers.** Four runs so far carried a
  provider, setting or `git_sha` that did not match intent. One recorded
  `overall_score: 0.0` on 232 consecutive HTTP 401s.
- **`hit@k` and `recall@k` are different numbers and this repo reports both.**
  `hit@5 = 0.6595` means at least one relevant chunk reached the top 5.
  `recall@5 = 0.2746` means 27% of all relevant chunks did. Quoting the first
  under the second's name is the single easiest way to get caught.

---

## 1. Retrieval quality

**Produce it**

```bash
edgar-intel eval retrieval                       # measured, below
edgar-intel eval compare --strategies fixed,recursive,section_aware,semantic
```

**Status: partly measured. The four-strategy comparison has never been run.**

Measured on the reference run, `section_aware` only:

| hit@1 | hit@3 | hit@5 | hit@10 | recall@5 | MRR | nDCG@10 | retrieve_ms |
|---|---|---|---|---|---|---|---|
| 0.3405 | 0.5862 | 0.6595 | 0.7414 | 0.2746 | 0.4738 | 0.3121 | 118 |

A separate configuration sweep (`reports/retrieval_sweep.json`, 232 cases,
15 Sep) compared hybrid / dense-only / lexical-only / rerank-on / wide-k and
found the candidate-set ceiling at **0.677 hit@5 against 0.483 shipped — 0.194
lost to ranking and `top_n=8` truncation, not to retrieval failing to find the
evidence.** That gap is the most useful retrieval finding in the project.

> ⚠️ **The sweep predates the evidence-link rewrite** (`f5232e6`), which changed
> the relevance labels every one of its numbers was scored against — the current
> set measures hit@5 0.6595 against the sweep's 0.483 for the shipped config, and
> most of that difference is the labels, not the retriever. Its *relative*
> ordering is still informative; its absolute numbers are not current. Re-run it
> before any of them go on a resume.

**Bullet shape — usable now**

> Measured hybrid retrieval (pgvector dense + Postgres full-text, RRF-fused) on
> a 232-case labelled set over 8 companies' SEC filings at hit@5 0.660 /
> MRR 0.474 / nDCG@10 0.312.

and, as a **separate** claim from a **separate** run — never fused into one
sentence, because the two were measured on different evidence labels:

> Diagnosed a 0.194 hit@5 gap between the shipped ranking and the candidate-set
> ceiling, showing the relevant passages were being retrieved and then discarded
> by `top_n` truncation rather than never found — which moved the work from
> prompt tuning to context selection.

**Bullet shape — blocked on `eval compare`, and `eval compare` is blocked**

> ~~Benchmarked four chunking strategies…~~ Do not write this until
> `eval compare` runs. All four strategies are chunked and fully embedded
> (23,499 chunks); none has been evaluated against the others.

> ⚠️ **Do not run the comparison yet.** `edgar-intel status` shows
> `section_aware` averaging **473 `token_est`** — and `token_est` is
> `len(body) // 4`, so about 1,892 characters. `all-MiniLM-L6-v2`'s sequence
> window is **256 word-piece tokens**, roughly 1,000–1,300 characters of English
> prose, and `LocalEmbedder.embed` never sets `max_seq_length`. So a substantial
> fraction of the average chunk — plausibly about half — is silently dropped
> before it is embedded. The same reasoning was applied carefully to the
> *reranker* (`rerank_max_length = 512`, with an explicit budget for `[CLS]` and
> two `[SEP]`, and a config comment about scoring a passage the model has only
> partly read) and never to the embedder.
>
> If that holds, all four strategies are crippled by the same truncation and the
> comparison would rank four handicapped runners. Worse, the ranking could invert
> once chunks fit the window — so the table would be a wrong finding, published.
>
> Settle it first by counting real tokens rather than estimating:
> ```python
> from sentence_transformers import SentenceTransformer
> m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
> print("window:", m.max_seq_length)
> # then tokenize 200 sampled section_aware chunk bodies and report the
> # fraction whose token count exceeds that window
> ```
> The fix is to match chunk size to the model's window (~250 `token_est`) rather
> than to an arbitrary 512, and to re-measure hit@5 — not to raise
> `max_seq_length`, which MiniLM was not trained at.

**Expect to be asked:** why hit@5 and not accuracy; what recall@k does not tell
you (nothing about ordering — that is what nDCG is for); why reranking can leave
recall unchanged while improving nDCG. And, on this project specifically: why
`use_rerank=false` shipped — the answer is that the reranker's verdict reversed
twice across runs and cost 525 ms of a 539 ms retrieval budget for a margin
inside noise, so it was switched off pending a clean single-variable run.

---

## 2. The eval set itself

**Produce it**

```bash
edgar-intel eval build --numeric 20 --narrative 3
edgar-intel ingest verify-facts          # the check that makes it trustworthy
```

**Status: measured.** 232 cases, all 232 with linked evidence.

**Bullet shape**

> Built an evaluation harness of **232** cases — **208** numeric questions
> generated from SEC XBRL company facts with machine-verifiable answers (160
> single-hop, 48 comparative) and **24** hand-written narrative questions with
> filing-grounded reference answers — giving the system a quality baseline that
> does not depend on an LLM grading another LLM.

**The stronger version of this bullet is about the ground truth, not the size:**

> Found and fixed three defects in the evaluation harness itself — fiscal years
> attributed from the filing rather than the reported period, the grader parsing
> the question's year as the answer, and abstentions scored as hallucinations —
> and added an abort on infrastructure failure after a run of 232 consecutive
> HTTP 401s was recorded as a score of 0.0 rather than as a void run.

That is the bullet an interviewer remembers, because almost nobody has debugged
their own ground truth.

> ⚠️ **Do not write "accuracy 0.25 → 0.62".** That comparison does not survive
> checking, and it is currently in the evidence log:
>
> | run | n | provider | numeric accuracy | |
> |---|---|---|---|---|
> | `smoke20-011be57c` | **20** | openai | 0.25 | the source of the 0.25 |
> | `baseline-45536dc9` | 232 | openai | 0.0 | void — 401 on every case |
> | `baseline-f6da95f6` | 232 | openai | 0.0 | void |
> | `baseline-c8d68dd6` | 232 | **fake** | 0.0337 | not a real model |
> | `baseline-d2f33b7b` | 232 | openai | 0.5625 | first trustworthy run |
> | `baseline-v5-2943b37a` | 232 | openai | **0.6202** | current |
>
> The 0.25 is a **20-case smoke run**, not a baseline. And the deeper problem is
> that the defects being fixed *changed the golden set itself* — corrected fiscal
> years, then `PREFERRED_TAG_FAMILIES` changing which XBRL concept a case asks
> about. **An accuracy delta that straddles a ground-truth rebuild compares two
> different exams.** The defensible claims are the absolute current figure
> (0.6202 on 232 cases) and the defects found — not a delta.

Saying that out loud in an interview is worth more than the delta would have
been. It is the difference between someone who reports numbers and someone who
knows which numbers are comparable.

**Why this section matters most:** evals appear in roughly 68% of AI-company
postings and almost no early-career candidate has built one.

---

## 3. Judge calibration

**Produce it**

```bash
edgar-intel eval run --label baseline
edgar-intel eval label <run_key> --n 40      # ~30 minutes of your time
edgar-intel eval judge-kappa --contracts v2,v3_1 --repeats 9
```

**Status: measured, and the result is the opposite of what this section
originally assumed.**

24 human-labelled narrative pairs, 3 repeats per cell, 17 Sep
(`reports/judge_kappa_v2_v3_v31.json`, `reports/judge_kappa_context_v2_vs_v31.json`):

| condition | contract | κ (per pair) | 95% CI | agreement | FP | FN |
|---|---|---|---|---|---|---|
| judge sees source context *(the shipped config)* | v2 | **0.4167** | [0.12, 0.74] | 17/24 | 7 | 0 |
| judge sees source context | v3_1 | 0.4167 | [0.12, 0.74] | 17/24 | 7 | 0 |
| judge context-free | v2 | 0.5833 | [0.24, 0.90] | 19/24 | 4 | 1 |
| judge context-free | v3 | 0.6667 | [0.33, 0.92] | 20/24 | 1 | 3 |
| judge context-free | v3_1 | 0.6667 | [0.33, 0.92] | 20/24 | 1 | 3 |

Four findings, none of which the original template anticipated:

1. **κ is below the 0.60 floor in the configuration that shipped.** So the
   floor did its job: `baseline-v5` reports `narrative_pass_rate 0.7917` and the
   human labels say **12/24 = 0.50**. The judge's number is not usable and is
   not quoted anywhere as if it were.
2. **The errors are perfectly one-sided** — 7 false positives, 0 false
   negatives. A permissive judge, not a noisy one, so the fix is a missing rule
   rather than a bigger model.
3. **Giving the judge the source passages made it worse** (κ 0.58 → 0.42, FP
   4 → 7). With passages to check against it stopped failing anything. That is a
   counter-intuitive result and it is the most interesting thing here.
4. **Three rubric revisions bought nothing in that condition.** v2 and v3_1
   produce byte-identical confusion matrices with context. Context-free, v3
   trades 3 false positives for 9 false negatives — and the adoption gate
   **blocked** it, because a stricter rubric that starts failing correct answers
   is the v1 defect returning, whatever it did to κ.

**Bullet shape**

> Calibrated an LLM-as-judge against 24 human-labelled answers, measuring
> Cohen's κ with bootstrap confidence intervals and gating narrative scores on a
> κ ≥ 0.60 floor — the judge came back at **κ = 0.42 with 7 false positives and
> 0 false negatives**, so the gate suppressed a reported 79% pass rate that was
> really 50%, and three rubric revisions failed to move it.

**Do not** write "κ = 0.67, substantial agreement." It is true of one arm of one
experiment in a condition the system does not run in, and it is the arm the
adoption gate rejected.

**Expect to be asked:** what κ measures that raw agreement does not (70.8%
agreement, κ 0.42 — half the agreement was chance); what you would do if κ came
back at 0.3. You have a better answer than most: *I measured it, it came back
unusable, I said so in the report instead of shipping the number.*

**Not yet run:** the 9-repeat instrument with duplicate control cells and the
reference-last prompt structure (committed at `236b4bf`). Everything above is
3 repeats. Until the 9-repeat run exists, run-to-run judge variance is bounded
only loosely — v2 context-free moved 0.5556 → 0.5278 between two nominally
identical runs, with 2 pairs flagged unstable.

---

## 4. Regression gate in CI

**Produce it**

```bash
edgar-intel eval gate --max-regression 0.03
```

**Status: built, never yet caught anything. Do not claim a catch count.**

**Bullet shape — usable now**

> Wired the evaluation suite into a regression gate that blocks on any drop
> above 3 points in overall score, numeric accuracy or recall@5, and a separate
> judge-adoption gate that vetoes a new judge rubric on a single new false
> negative regardless of its κ.

The second clause is the better half and it *has* fired — it blocked v3_1.

---

## 5. Failure attribution

**Produce it**

```bash
edgar-intel eval failures <run_key>
edgar-intel eval context-probe --case-id <case>
```

**Status: measured.** 84 failures on the reference run — 79 numeric, 5
narrative by the judge (12 narrative by human label). Attributed:

| | |
|---|---|
| retrieval misses — evidence never reached the model | **52 / 84 (62%)** |
| generation misses — evidence was present, answer still wrong | 32 / 84 (38%) |
| unlabelled | 0 |

- `abstention_rate` **0.1587** and `hallucination_rate` **0.2212**, split apart
  so a refusal is no longer counted as a fabrication.
- `hit@5` 0.6595, so roughly a third of cases have no relevant evidence in the
  top 8 — which is where the abstentions live.

**Bullet shape**

> Instrumented per-case retrieval metrics to separate retrieval misses from
> generation misses, and split abstention from hallucination as distinct
> outcomes (0.159 / 0.221) — attributing **62% of failures (52 of 84) to
> evidence that never reached the model**, which redirected the work from prompt
> tuning to context selection.

That 62% is the harness's own attribution, and it is the number that justifies
every retrieval decision in the project. It is also why the embedder-window
question below has to be settled before the chunking comparison runs.

The context probe is the sharper story: for one case all five relevant chunks
sat in the hybrid top-50 at ranks 9, 11, 13, 21 and 29, and `top_n=8` discarded
every one. The run reported a quality failure. It was a truncation failure.

---

## 6. Agent reliability

**Produce it**

```bash
edgar-intel agent stats
```

**Status: NOT MEASURED. `agent_traces` is 0 — the agent has never been run
outside tests. Do not claim numbers, and expect `/stats/agent` to come back
empty on a deployed instance.**

**Bullet shape — mechanism only, which is defensible today**

> Engineered a bounded tool-calling agent over 6 typed tools with an 8-step
> ceiling, retry-with-backoff, loop detection and a confidence-floor escalation
> path.

Fill in the run count, answered % and escalated % only after `agent stats`
returns them.

**Expect to be asked:** what your escalation rate is and whether it is right. A
0% escalation rate is not a good sign — it usually means the confidence floor is
too low and the agent is answering things it should hand off.

---

## 7. Hallucination prevention

**Status: mechanism shipped. No number needed.**

**Bullet shape**

> Added citation and numeric-grounding validation that rejects any answer citing
> a passage no tool returned, or quoting a figure absent from every tool output,
> converting hallucination detection from a model judgement into a code-level
> check.

---

## 8. Latency and cost

**Produce it**

```bash
edgar-intel bench sweep --endpoint /search     # concurrency sweep: NOT RUN
```

**Status: end-to-end latency measured; the concurrency sweep has not been run.**

From the reference run: **p50 996 ms, p95 1700 ms** end to end (retrieval 118 ms
of it), at **$0.126 for the full 232-case run** and **$0.5435 per 1k requests**.

**Bullet shape — usable now**

> Instrumented per-request token accounting and stage-level timing across
> retrieval, generation and judging: p50 996 ms / p95 1700 ms end to end at
> $0.13 per full 232-case evaluation and $0.54 per 1k requests.

**Bullet shape — blocked**

> ~~Benchmarked the retrieval API across concurrency levels 1–32…~~ Requires a
> deployed instance and `bench sweep`. Not yet run.

**Expect to be asked:** why p95 and not mean (a service averaging 400 ms with a
p99 of nine seconds is visibly broken for one user in a hundred, and the mean
never tells you).

---

## 9. Fine-tuning tradeoff

**Status: NOT MEASURED. Code exists; no training run, no benchmark.**

> Fine-tuned `<model>` with LoRA on `<N>` distantly-supervised extraction
> examples, lifting F1 from `<baseline>` (few-shot prompting) to `<result>` on a
> company-disjoint held-out set at `<X>`× lower cost per request, against
> `<H>` GPU-hours of training.

**The comparison is the bullet, not the training.** Placeholders left
deliberately — this one is honestly unmeasured and is the lowest-priority item
in the repo.

---

## Scoreboard — what can go on a resume today

| claim | status |
|---|---|
| 232-case eval set with XBRL-verifiable ground truth | ✅ measured |
| numeric accuracy **0.6202** on the current set | ✅ measured |
| "accuracy 0.25 → 0.62" | ❌ **not defensible** — 20-case smoke vs 232-case baseline, across a ground-truth rebuild |
| three ground-truth defects found and fixed | ✅ measured |
| void-run abort after 232 consecutive 401s scored as 0.0 | ✅ measured |
| hit@5 0.660 / MRR 0.474, and the 0.194 ceiling gap | ✅ measured |
| abstention 0.159 vs hallucination 0.221, split | ✅ measured |
| 62% of failures (52/84) attributed to retrieval, not generation | ✅ measured |
| corpus: 8 companies, 24 10-K filings, 264 sections, 23,499 chunks | ✅ measured |
| p50 996 ms / p95 1700 ms, $0.54 per 1k | ✅ measured |
| Cohen's κ = 0.42, below the floor, gate suppressed a wrong 79% | ✅ measured |
| judge-adoption gate vetoed a rubric on one false negative | ✅ measured |
| bounded agent: mechanism | ✅ shipped |
| narrative pass rate | ⚠️ 0.50 by human label on 24 cases; the judge's 0.79 is not usable |
| four-strategy chunking comparison | ⛔ **blocked** — settle the embedder window first, or the table is a wrong finding |
| agent escalation rate | ❌ never run |
| concurrency / throughput sweep | ❌ needs a deployed instance |
| regression-gate catch count | ❌ nothing caught yet |
| LoRA fine-tune comparison | ❌ never run |
| live URL | ❌ not deployed — see `docs/DEPLOY.md` |

## Metrics that carry the most weight, ranked

1. **recall@k or hit@k with a baseline** — rare on early-career resumes,
   immediately credible
2. **Cohen's κ for judge calibration** — almost nobody has this, and a κ that
   came back *bad* plus the gate that caught it is stronger than a good κ
3. **p95 latency** — signals you have operated a service, not just built one
4. **$/1k requests** — signals you think about constraints
5. **Eval-set size with provenance** — "232 cases, generated from XBRL"
6. **Regression-gate catches** — signals production discipline
7. **Escalation rate** — signals you understand agent reliability

## Metrics to avoid

- **Accuracy with no baseline.** Unfalsifiable, and reads as such.
- **Mean latency.** Hides the behaviour that matters.
- **"Improved by 20%"** with no starting point. Invites the one question you
  cannot answer.
- **A judge's own pass rate, when κ says the judge is not calibrated.** This
  repo reports 0.7917 and the truth is 0.50. The report says so; a resume that
  quoted the 0.79 would not.
- **Anything you have not personally measured.** Every number on a resume is a
  promise to explain it under questioning.
