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
| Reference run | `baseline-v6-no-rerank-7362c6c7`, 26 Sep 2026, `git_sha` `cc0a69c`, clean tree |
| Report | `reports/baseline-v6-no-rerank-7362c6c7.json` |
| Same-commit control | `baseline-v6-rerank-8d0eb06a` — identical config except `use_rerank=true` |
| Re-grade of the previous reference | `reports/regrade-baseline-v5-2943b37a.json` — `baseline-v5`'s stored answers re-marked by the fixed grader, no model calls |
| Set | 232 cases — 160 numeric single-hop · 48 numeric comparative · 24 narrative |
| Corpus | 8 companies (AAPL CAT COST JNJ MSFT NVDA PG UNH), **24 10-K filings**, FY2023–FY2026 · 264 sections · 1,338 XBRL facts · 23,499 chunks across four strategies, all embedded — `section_aware` 5,468 is the evaluated and served index. (A 32,171-chunk `section_aware_t144` experiment also exists locally; it is not shipped — §1.) |
| Config | `section_aware` · `hybrid` · `use_rerank=false` · `k=50` · `top_n=8` · `gpt-4o-mini` · MiniLM-L6-v2 (384-d) · `rrf_k=60` · `numeric_tolerance=0.005` · `context_max_chars=12000` · judge contract `v2`, shown the source context |

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
edgar-intel eval retrieval --out reports/retrieval_<date>.json   # measured, below
edgar-intel eval compare --strategies fixed,recursive,section_aware,semantic
```

**Status: partly measured. The four-strategy comparison has never been run.**

Measured on the reference run, `section_aware` only — identical to `baseline-v5`
to four decimal places, as it should be: same index, same labels, same retriever:

| hit@1 | hit@3 | hit@5 | hit@10 | recall@5 | MRR | nDCG@10 | retrieve_ms |
|---|---|---|---|---|---|---|---|
| 0.3405 | 0.5862 | 0.6595 | 0.7414 | 0.2746 | 0.4738 | 0.3121 | 105 |

The configuration sweep was re-run on the current labels on 26 Sep
(`reports/retrieval_current_d7.json`; all 490 labels present in the index). The
untruncated candidate set reaches **0.853 hit@20 against 0.660 hit@5 for the
shipped configuration — 0.194 lost to ranking and `top_n=8` truncation, not to
retrieval failing to find the evidence.** Dense-only reaches 0.810 and
lexical-only 0.836 at the same depth. The 15 Sep sweep, on the pre-`f5232e6`
labels and with the reranked row then called "shipped", had also put the gap at
0.194 (0.483 against 0.677); the absolute figures moved with the labels.

| config (26 Sep sweep) | rerank | hit@1 | hit@5 | recall@5 | MRR | nDCG@10 | ms |
|---|---|---|---|---|---|---|---|
| shipped | off | 0.3405 | **0.6595** | 0.2746 | **0.4738** | 0.3121 | 19 |
| same, reranked | on | 0.3448 | 0.6379 | 0.2759 | 0.4700 | 0.3176 | 558 |

(The sweep file labels the reranked row "shipped" and the other "no-rerank": it
was produced before the sweep learned to read `use_rerank`. The rows above are
named for what they ran.)

**Bullet shape — usable now**

> Measured hybrid retrieval (pgvector dense + Postgres full-text, RRF-fused) on
> a 232-case labelled set over 8 companies' SEC filings at hit@5 0.660 /
> MRR 0.474 / nDCG@10 0.312.

and the ceiling gap, now measured on the same labels as the reference run:

> Diagnosed a 0.194 hit@5 gap between the shipped ranking and the candidate-set
> ceiling, showing the relevant passages were being retrieved and then discarded
> by `top_n` truncation rather than never found — which moved the work from
> prompt tuning to context selection.

**Bullet shape — blocked on `eval compare`, and `eval compare` is blocked**

> ~~Benchmarked four chunking strategies…~~ Do not write this until
> `eval compare` runs. All four strategies are chunked and fully embedded
> (23,499 chunks); none has been evaluated against the others.

> ⚠️ **The embedder window is still unsettled (D8).** `all-MiniLM-L6-v2` reads 256
> word-pieces and `section_aware` chunks average 473 `token_est`, and
> `edgar-intel index embed-window` confirmed truncation: it put the ceiling that
> fits the window at **144 `token_est`**. Re-chunking to that ceiling
> (`section_aware_t144`, 32,171 chunks, `reports/retrieval_section_aware_t144_d8.json`)
> made every figure worse — hit@5 0.660 → 0.330, recall@5 0.275 → 0.104, MRR
> 0.474 → 0.217 — **and lexical-only fell as far as dense-only** (0.836 → 0.552 at
> depth), which the embedder's window cannot touch. So that experiment measured
> chunk granularity and re-derived labels, not truncation; it is not shipped.
>
> The clean test keeps the chunks and their labels fixed and changes only the
> window: the same `section_aware` chunks re-embedded, beside the current index
> (`index build --suffix`), by a 384-d model that reads 512 tokens. Until then the
> four-strategy comparison would still rank four runners handicapped the same
> way. Two fixes it will need are already in: every strategy now treats
> `target_tokens` as a ceiling (D9), and each arm is graded on labels derived for
> its own index (D11) — without which three of the four would have scored
> hit@k = 0 by construction.

**Expect to be asked:** why hit@5 and not accuracy; what recall@k does not tell
you (nothing about ordering — that is what nDCG is for); why reranking can leave
recall unchanged while improving nDCG. And, on this project specifically: why
`use_rerank=false` ships — the answer is one run at one commit changing only the
flag: the cross-encoder added **542 ms at p50** (892 → 1434 ms), numeric accuracy
went 0.7404 → 0.7356 and hit@5 0.660 → 0.638. It had reversed twice across
earlier runs that changed other things too; this is the first clean comparison.

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

A fourth, found 26 Sep and measured without a model call: **the comparative
grader failed correct answers.** "How did X change from FY2023 to FY2024?" never
asks for a percentage; the model answered with both figures and was failed for
"no percentage found". Re-marking `baseline-v5`'s stored answers with the fixed
grader flipped **28 answers from fail to pass and none the other way** —
comparative cases went from 5/48 to 33/48, single-hop was untouched at 124/160.
Every flip is listed in `reports/regrade-baseline-v5-2943b37a.json`, and each
states both XBRL values exactly.

**Known and not yet fixed (D12):** two NVIDIA year-over-year cases compare across
the June 2024 10-for-1 split, because each year's value comes from its own
earliest filing — "EPS decreased 75.4%" (11.93 → 2.94) and "shares increased
893.4%" (2,464M → 24,477M). The model's split-adjusted answers are the right
ones. They are fixed with the next golden-set rebuild, not by re-grading.

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
> | `baseline-v5-2943b37a` | 232 | openai | 0.6202 | as graded on 17 Sep |
> | `baseline-v5-2943b37a`, re-graded | 232 | — | 0.7548 | same answers, D2 grader fix |
> | `baseline-v6-no-rerank-7362c6c7` | 232 | openai | **0.7404** | current |
>
> The 0.25 is a **20-case smoke run**, not a baseline. And the deeper problem is
> that the defects being fixed *changed the golden set itself* — corrected fiscal
> years, then `PREFERRED_TAG_FAMILIES` changing which XBRL concept a case asks
> about. **An accuracy delta that straddles a ground-truth rebuild compares two
> different exams.** The defensible claims are the absolute current figure
> (**0.7404** on 232 cases) and the defects found.
>
> The one delta that *is* quotable is the re-grade, because nothing else moved:
> same stored answers, same golden set, same tolerance, only the grading code —
> **0.6202 → 0.7548**, with every flip listed. `baseline-v6` then came within three
> cases of it on freshly generated answers (0.7404); regenerated answers move a
> few cases between runs even at temperature 0, so quote v6 as the current figure
> and the re-grade as the size of the grader fix.

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

The same holds for `baseline-v6`: its judge reports 0.75, only 12 of its answers
match a human label (the floor is 20), and no kappa is reported for it. The
deployed `/stats/eval` therefore **withholds** `narrative_pass_rate` and
`overall_score` and prints the kappa verdict in their place; the stored run keeps
every value.

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

**Status: built, never yet caught anything — and not running in CI. Do not claim
a catch count, and do not call it a CI gate until the job runs.**

> ⚠️ The `eval-gate` job in `.github/workflows/ci.yml` cannot pass as written:
> `evalset/` is gitignored, so the checkout has no `golden.json` for `eval run`
> to load, and the job applies only `sql/001_schema.sql`, so the columns
> `sql/003_…` adds — which every run writes — do not exist. Separately,
> `ruff check src tests` reports 16 findings on the current code, so the `test`
> job most likely stops at lint.

**Bullet shape — usable now**

> Wired the evaluation suite into a regression gate that blocks on any drop
> above 3 points in overall score, numeric accuracy or recall@5, and a separate
> judge-adoption gate that vetoes a new judge rubric on a single new false
> negative regardless of its κ.

Say "a gate command" rather than "CI" until the job above passes. The second
clause is the better half and it *has* fired — it blocked v3_1.

---

## 5. Failure attribution

**Produce it**

```bash
edgar-intel eval failures <run_key>
edgar-intel eval context-probe --case-id <case>
```

**Status: measured on `baseline-v5`'s answers, before and after the grader fix
(D2). For the current run, `edgar-intel eval failures baseline-v6-no-rerank-7362c6c7`
has not been run yet.**

| `baseline-v5` answers | as graded, 17 Sep | re-graded, 26 Sep |
|---|---|---|
| failures | 84 (79 numeric, 5 narrative by the judge) | 56 (51 numeric, 5 narrative) |
| retrieval misses — nothing labelled in the top 5 | 52 (62%) | **46 (82%)** |
| generation misses — labelled evidence in the top 5, answer still wrong | 32 (38%) | **10 (18%)** |
| unlabelled | 0 | 0 |

The grader fix moved this more than any other figure: 22 of the 28 re-graded
answers had been counted as *generation* misses — correct answers the grader could
not read. The other 6 had been counted as *retrieval* misses, correct answers with
nothing labelled in the top 5. That second group shows the rule over-counts
retrieval: the model reads up to 8 passages, and the labels are a sample of the
relevant chunks rather than all of them. Quote the direction, not the decimal.

- `abstention_rate` / `hallucination_rate`: **0.1635 / 0.0962** on `baseline-v6`.
  `baseline-v5` read 0.1587 / 0.2212 as graded and 0.1587 / 0.0865 re-graded —
  every D2 answer had been sitting in the hallucination bucket.
- `hit@5` 0.6595, so roughly a third of cases have no relevant evidence in the
  top 5 — which is where the abstentions live.

**Bullet shape**

> Instrumented per-case retrieval metrics to separate retrieval misses from
> generation misses, and split abstention from hallucination as distinct
> outcomes (0.164 / 0.096) — after fixing a grader that had failed 28 correct
> answers, most remaining failures (46 of 56) had no labelled evidence in the
> retrieved top 5, which pointed the work at retrieval and context selection
> rather than the prompt.

It is the harness's own attribution, and it is why the embedder-window question
(§1) has to be settled before the chunking comparison runs.

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

**Status: end-to-end latency measured locally; the concurrency sweep has not been
run.**

From the reference run: **p50 892 ms, p95 1732 ms** end to end (retrieval 105 ms
of it), at **$0.126 for the full 232-case run** and **$0.5439 per 1k requests**.
The same-commit reranked run: p50 1434 ms, p95 2340 ms, $0.5582 per 1k — the
cross-encoder's 542 ms at p50 is the price that bought nothing (§1).

**Bullet shape — usable now**

> Instrumented per-request token accounting and stage-level timing across
> retrieval, generation and judging: p50 892 ms / p95 1732 ms end to end at
> $0.13 per full 232-case evaluation and $0.54 per 1k requests.

These are local figures on a warm process. A deployed instance's numbers come
from `bench sweep` against its URL, not from here.

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
| numeric accuracy **0.7404** on the current set (`baseline-v6`) | ✅ measured |
| grader fix re-graded the same answers **0.6202 → 0.7548**, 28 flips listed, 0 reversed | ✅ measured |
| "accuracy 0.25 → 0.62" | ❌ **not defensible** — 20-case smoke vs 232-case baseline, across a ground-truth rebuild |
| four ground-truth and grading defects found and fixed | ✅ measured |
| void-run abort after 232 consecutive 401s scored as 0.0 | ✅ measured |
| hit@5 0.660 / MRR 0.474, and the 0.194 ceiling gap (re-measured on current labels) | ✅ measured |
| reranker off on a single-variable run: +542 ms p50 for no gain | ✅ measured |
| abstention 0.164 vs hallucination 0.096, split | ✅ measured |
| most remaining failures (46 of 56) had no labelled evidence in the top 5 | ⚠️ measured on re-graded v5; the rule over-counts retrieval — quote the direction |
| corpus: 8 companies, 24 10-K filings, 264 sections, 23,499 chunks | ✅ measured |
| p50 892 ms / p95 1732 ms, $0.54 per 1k — local | ✅ measured |
| Cohen's κ = 0.42, below the floor, gate suppressed a wrong 79% | ✅ measured |
| judge-adoption gate vetoed a rubric on one false negative | ✅ measured |
| bounded agent: mechanism | ✅ shipped |
| narrative pass rate | ⚠️ 0.50 by human label on 24 cases; the judge's 0.79 (v5) and 0.75 (v6) are not usable |
| two NVIDIA yoy cases compare across a stock split (D12) | ⚠️ known; fixed at the next golden-set rebuild |
| four-strategy chunking comparison | ⛔ **blocked** — the embedder-window test that isolates truncation has not run (§1) |
| agent escalation rate | ❌ never run |
| concurrency / throughput sweep | ❌ needs a deployed instance |
| regression gate in CI | ❌ the CI job cannot pass as written (§4); nothing caught yet |
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
  repo reported 0.7917 and the truth was 0.50. The report says so, and the
  deployed `/stats/eval` withholds the figure; a resume that quoted it would not.
- **Anything you have not personally measured.** Every number on a resume is a
  promise to explain it under questioning.
