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
| Reference run | `exp-prompt-conventions-c0f6e105`, 6 Oct 2026, `git_sha` `bee5dc0`, clean tree: the shipped configuration with the answering prompt's financial-reading conventions (§5), on the D15 labels |
| Report | `reports/exp-prompt-conventions-c0f6e105.json` |
| Prompt control | `baseline-v8-7ae05834`, 6 Oct, `c152a61`: an identical config with the earlier answering prompt. The run config does not record the prompt, so the `git_sha` is the record. |
| Same-config re-run | `exp-ctx20k-834a873e`, 5 Oct, `0e96e20`: the context-cap experiment's arm, with the same config and prompt as `baseline-v8`. 175/208 against v8's 174/208, so read 1–2 cases as run-to-run noise. |
| Context-cap control | `baseline-v7-99204312`, 5 Oct, `47251c1`: the same set, index and config with the old 12,000-character cap, against `exp-ctx20k`. The single-variable comparison in §5. |
| Agent runs (§6) | `agent-as-deployed-7a4e5c17`, 6 Oct, `274419a`: the tool-calling agent `/ask` serves, on the same set and grader. `agent-citation-fix-c3da771e` (`c4a7424`) differs only in `get_fact`'s printed citation; `agent-citation-fix-dns-retry-387b41f9` re-ran the 7 cases a DNS outage errored in it; `agent-citation-fix-rerun-65631f87` (`6451ac4`) repeats it unchanged to measure noise. All carry `pipeline: agent`, so `/stats/eval` does not serve them. |
| Pre-rebuild reference | `baseline-v6-no-rerank-7362c6c7`, 26 Sep, `cc0a69c`, on the pre-rebuild set (archived locally as `evalset/golden-baseline-v6.json`). Same config as v7. It graded a different exam, and §2 shows exactly where the two differ. |
| Rerank control | `baseline-v6-rerank-8d0eb06a`: v6's config with `use_rerank=true`, on the v6 set. Not re-run on the v7 set. |
| Re-grade of an earlier reference | `reports/regrade-baseline-v5-2943b37a.json`: `baseline-v5`'s stored answers re-marked by the fixed grader, no model calls |
| Set | 232 cases — 160 numeric single-hop · 48 numeric comparative · 24 narrative. Rebuilt 5 Oct with the D12–D14 fixes, then re-linked for D15 the same day (labels only; the exam is unchanged), 231 with linked evidence |
| Corpus | 8 companies (AAPL CAT COST JNJ MSFT NVDA PG UNH), **24 10-K filings**, FY2023–FY2026 · 264 sections · 1,363 XBRL facts (1,338 before the D13 refresh) · 23,499 chunks across four strategies, all embedded — `section_aware` 5,468 is the evaluated and served index. (A 32,171-chunk `section_aware_t144` experiment also exists locally; it is not shipped — §1.) |
| Config | `section_aware` · `hybrid` · `use_rerank=false` · `k=50` · `top_n=8` · `gpt-4o-mini` · MiniLM-L6-v2 (384-d) · `rrf_k=60` · `numeric_tolerance=0.005` · `context_max_chars=20000` · answering prompt at `bee5dc0` · judge contract `v2`, shown the source context |

Two standing cautions, both earned the hard way:

- **Outside §6, the evaluation measures retrieve-then-answer, not `/ask`.**
  Every accuracy figure in §1–§5 comes from `eval run`: hybrid retrieval, then
  one answering call with `ANSWER_SYSTEM`. The deployed `/ask` runs the
  tool-calling agent, with its own prompts and direct XBRL tools. §6 measures
  it on the same set with the same grader (`eval agent`), and it scores
  differently: **0.7115** numeric accuracy after a citation fix, 0.4135 as
  deployed, against 0.8606. Never quote one pipeline's figure as the other's.
- **Read the `config` block before the numbers.** Four runs so far carried a
  provider, setting or `git_sha` that did not match intent. One recorded
  `overall_score: 0.0` on 232 consecutive HTTP 401s.
- **`hit@k` and `recall@k` are different numbers and this repo reports both.**
  `hit@5 = 0.7619` means at least one relevant chunk reached the top 5.
  `recall@5 = 0.4084` means 41% of all relevant chunks did. Quoting the first
  under the second's name is the single easiest way to get caught.

---

## 1. Retrieval quality

**Produce it**

```bash
edgar-intel eval retrieval --out reports/retrieval_<date>.json   # measured, below
edgar-intel index build --strategy all --suffix _d9                # rebuild beside, never in place
edgar-intel eval compare --strategies fixed_d9,recursive_d9,section_aware_d9,semantic_d9
```

**Status: measured, including the four-strategy comparison (6 Oct).**

Measured on the reference run, `section_aware` only, on the rebuilt labels
(741 label references to 396 chunks on 231 cases, every one reachable since
D15). `top_n=8`, so hit@10 here
is hit@8:

| hit@1 | hit@3 | hit@5 | hit@10 | recall@5 | MRR | nDCG@10 | retrieve_ms |
|---|---|---|---|---|---|---|---|
| 0.4026 | 0.6883 | 0.7619 | 0.8528 | 0.4084 | 0.5552 | 0.4288 | 134 |

The hit@k and MRR figures are identical in `baseline-v7` (apart from timing),
because raising the context cap changes what the model reads, not what
retrieval returns. The reference run was made after D15, so its stored summary,
which a deployed `/stats/eval` serves, carries these figures, recall@5 0.408
included. The free D15 sweep (`reports/retrieval_d15_20261005.json`) gives the
same numbers.

**D15, fixed 5 Oct: labels search could never reach.** 178 of the 919 label
references (19%) pointed at chunks in a filing other than the case's fiscal
year. Search filters on that year, so those chunks could never be retrieved,
yet recall@k counted each as a miss. The linker now labels only the filing
search reaches. On the same retrieval results recall@5 goes **0.320 → 0.408**
and nDCG@10 0.367 → 0.429, while hit@k, MRR and the ceiling gap do not move.
That is the signature of a measurement fix, not a retrieval change.

The rise over `baseline-v6` (hit@5 0.6595, MRR 0.4738) is the labels, not the
retriever. Both runs used the same index and the same retrieval behaviour, and
on the 194 cases whose labels did not change, hit@5 is 0.7577 in both. The
difference sits in the 38 relabelled cases, 26 of them EPS cases whose old
labels were coincidental (D14, §2). Quote 0.762 as the current figure, never
"0.660 → 0.762".

The configuration sweep was re-run on the rebuilt labels on 5 Oct
(`reports/retrieval_v7_20261005.json`; on the D15 labels,
`reports/retrieval_d15_20261005.json`; no model calls). The untruncated
candidate set reaches **0.965 hit@20 against 0.762 hit@5 for the shipped
configuration — 0.204 lost to ranking and `top_n=8` truncation, not to
retrieval failing to find the evidence.** Dense-only reaches 0.922 and
lexical-only 0.939 at the same depth. On the v6 labels (26 Sep,
`reports/retrieval_current_d7.json`) the gap was 0.194 (0.853 against 0.660),
and on the 15 Sep labels it was also 0.194. The absolute figures moved with the
labels each time, but the gap barely did.

| config (5 Oct sweep, D15 labels) | rerank | hit@1 | hit@5 | recall@5 | MRR | nDCG@10 | ms |
|---|---|---|---|---|---|---|---|
| shipped | off | 0.4026 | **0.7619** | 0.4084 | **0.5552** | 0.4288 | 21 |
| same, reranked | on | 0.4113 | 0.7489 | 0.4381 | 0.5550 | 0.4535 | 553 |

(The 26 Sep sweep file labels its reranked row "shipped", because it was
produced before the sweep learned to read `use_rerank`. The 5 Oct file names
both rows for what they ran.)

**Bullet shape — usable now**

> Measured hybrid retrieval (pgvector dense + Postgres full-text, RRF-fused) on
> a 232-case labelled set over 8 companies' SEC filings at hit@5 0.762 /
> MRR 0.555 / nDCG@10 0.429.

and the ceiling gap, measured on the same labels as the reference run:

> Diagnosed a 0.204 hit@5 gap between the shipped ranking and the candidate-set
> ceiling, showing the relevant passages were being retrieved and then discarded
> by `top_n` truncation rather than never found — which moved the work from
> prompt tuning to context selection.

### The four-strategy comparison: the strategy mattered less than the chunk size

The four original indexes all date from 15 Sep, before D9 made `target_tokens`
a ceiling. `semantic` held one chunk of 31,476 `token_est`, about 125,000
characters, more than any context can hold, so comparing those builds would
have measured that bug. All four were rebuilt beside the originals with
today's chunker (`--suffix _d9`; 512 ceiling, 64 overlap, MiniLM) and run
through `eval compare` against the shipped index, with labels re-linked per
index (D11). The config was identical to the reference apart from the index,
and run-to-run noise is 1–2 cases:

| index | numeric accuracy | single-hop | comparative | hit@5 | mean prompt tokens | vs shipped |
|---|---|---|---|---|---|---|
| **`section_aware`, shipped (pre-D9)** | **0.837** | 136/160 | 38/48 | 0.762 | 4,828 | — |
| `recursive_d9` | 0.789 | 129/160 | 35/48 | 0.701 | 4,463 | +9 / −20 |
| `fixed_d9` | 0.784 | 121/160 | **42/48** | 0.693 | 4,889 | +18 / −33 |
| `section_aware_d9` | 0.784 | 127/160 | 36/48 | 0.710 | 4,465 | +8 / −21 |
| `semantic_d9` | 0.745 | 123/160 | 32/48 | 0.652 | 3,977 | +15 / −35 |

**On equal footing, three strategies tie.** Fixed, recursive and section-aware
chunking land within one case of each other. Knowing 10-K structure bought
nothing once every chunk was capped at the same size. **Semantic trails by 8–9
cases.** Its chunks average half the size (265 `token_est`), so at a fixed
`top_n=8` the model reads about 18% less text, which is part of the reason.
Fixed chunking's 42/48 on comparisons is 4 cases above the rest, a lead that
might be real but is close to the noise.

**The shipped index beats all four by 10–11 cases, and size is most of why.**
Built before D9, it added the section header and overlap *on top of* 512
tokens, so its chunks run about 15% larger, which keeps statement tables
together: nine of its rebuild's retrieval losses were liabilities. Rebuilding
`section_aware` under today's rules with a 600-token ceiling
(`exp-sa-d9t600-ccd9a5b3`; 4,668 chunks, average 2,148 characters, all eight
still inside the context cap) scored **0.817**. That is 7 cases above the
512-token rebuild, but still 4 below the shipped index, even though its chunks
are larger on average. So size explains most of the gap, and where the
boundaries fall explains the rest. The shipped index stays.

> Benchmarked four chunking strategies end to end on a 232-case set: at an
> equal 512-token ceiling, fixed, recursive and section-aware chunking tied
> (0.784–0.789 numeric accuracy) and semantic trailed (0.745). The bigger lever
> was chunk size: 512 → 600 tokens recovered 7 of the 11 cases separating the
> rebuild from the shipped index.

Because today's chunker does not reproduce the shipped index, `index build` now
refuses to replace an existing served index without `--replace-served`
(`48fcefd`). The README's own `index build --strategy all` would otherwise have
cost those 11 cases on the local database. A fresh database, such as CI or a
new deployment, builds as before. The rebuilt indexes were deleted after
measurement.

**D8, settled 5–6 Oct: the embedder window.** `all-MiniLM-L6-v2` reads 256
word-pieces, and 93% of `section_aware` chunks are longer than that (median
380, p90 875). In **51%** of the labelled numeric evidence, the expected figure
sits past word-piece 256, where the dense embedder never sees it; at a 512
window, 12% would be out of view. An earlier attempt re-chunked to fit the
window (`section_aware_t144`), which made every figure worse, lexical-only
included. That measured chunk granularity, not truncation, and is not shipped.

The clean test kept the chunks fixed. Every `section_aware` row was copied
byte-for-byte and embedded with `BAAI/bge-small-en-v1.5` (384-d), at a 256 and a
512 window. The control re-ran MiniLM afterwards, and its numbers were identical
to before. Free sweeps, on the D15 labels
(`reports/retrieval_d8_{minilm_control,bge256,bge512}_20261005.json`):

| shipped hybrid | hit@1 | hit@5 | recall@5 | MRR | nDCG@10 | dense-only hit@5 |
|---|---|---|---|---|---|---|
| MiniLM, 256 (shipped) | 0.403 | 0.762 | 0.408 | 0.555 | 0.429 | 0.701 |
| bge-small, 256 | 0.390 | 0.784 | 0.407 | 0.547 | 0.424 | 0.632 |
| bge-small, 512 | **0.489** | **0.805** | **0.437** | **0.615** | **0.469** | 0.701 |

**The window, not the model, carries the retrieval gain.** At 256, bge-small is
no better than MiniLM, and worse dense-only. Opening its window to 512 is what
lifts it. The mechanism shows on the cases D8 predicts: for the 36 whose figure
sits past word-piece 256 in every labelled chunk, dense hit@5 rises 0.361 →
0.556 between the two bge arms. Hybrid search hid most of that, because
keyword matching was already carrying those cases. The hybrid gain came from
the other numeric cases (0.772 → 0.848), while narrative fell 0.708 → 0.625.

**Then the paid run: retrieval improved, answers did not.**
`exp-bge512-e5726731` changed only the embedder and its index copy against the
`exp-ctx20k` run: numeric accuracy **0.8413 → 0.8365** (9 cases better, 10
worse), the uncalibrated narrative judge 0.875 → 0.708, and hallucination
0.048 → 0.063. The reference, a same-config re-run of the shipped embedder,
also scored **0.8365**: bge-512 tied it exactly, and 0.8413 sits within the
noise.
Of the 12 failing cases predicted beforehand to gain, 8 did, 7 of them cash
balances. But a new embedder reorders the passages for most questions, not
only the labelled ones, and the rest of the set lost more than it gained.
**bge-512 is not adopted, and the copies are deleted.** Truncation is real and
costs dense retrieval, but this fix does not pay for itself in answers. A
retriever fusing both embedders is untried.

> Isolated embedder truncation by re-embedding identical chunks at two windows:
> a 512-token window lifted hybrid hit@5 0.762 → 0.805 and MRR 0.555 → 0.615,
> but a single-variable paid run showed no accuracy gain (0.837, tying a
> same-config re-run of the shipped embedder), so the
> change was declined. Retrieval metrics over sampled labels did not predict
> answer quality.

With D8 settled, the four-strategy comparison ran (above). It depended on two
earlier fixes: every strategy treats `target_tokens` as a ceiling (D9), and each
arm is graded on labels derived for its own index (D11). Without D11, three of
the four would have scored hit@k = 0 by construction.

**Expect to be asked:** why hit@5 and not accuracy; what recall@k does not tell
you (nothing about ordering — that is what nDCG is for); why reranking can leave
recall unchanged while improving nDCG. And, on this project specifically: why
`use_rerank=false` ships — the answer is one run at one commit changing only the
flag: the cross-encoder added **542 ms at p50** (892 → 1434 ms), numeric accuracy
went 0.7404 → 0.7356 and hit@5 0.660 → 0.638. It had reversed twice across
earlier runs that changed other things too; this is the first clean comparison.
That was on the v6 set. On the rebuilt labels, the free sweep tells the same
story: reranking moves hit@5 0.762 → 0.749 and nDCG@10 0.429 → 0.454, for
about 530 ms. The paid single-variable run has not been repeated on the v7 set.

---

## 2. The eval set itself

**Produce it**

```bash
edgar-intel eval build --numeric 20 --narrative 3
edgar-intel ingest verify-facts          # the check that makes it trustworthy
```

**Status: measured.** 232 cases, 231 with linked evidence. Rebuilt 5 Oct 2026
on corrected facts. The one unlinked case is explained under D14.

**Bullet shape**

> Built an evaluation harness of **232** cases — **208** numeric questions
> generated from SEC XBRL company facts with machine-verifiable answers (160
> single-hop, 48 comparative) and **24** hand-written narrative questions with
> filing-grounded reference answers — giving the system a quality baseline that
> does not depend on an LLM grading another LLM.

**The stronger version of this bullet is about the ground truth, not the size:**

> Found and fixed eight defects in the evaluation harness itself, among them
> fiscal years attributed from the filing rather than the reported period, a
> 52/53-week year-end that filed a company's fiscal 2022 as 2023, year-over-year
> answers computed across a stock split, a grader that failed correct
> comparative answers, evidence labels matched on a single rounded digit, and
> labels placed in filings retrieval was never allowed to search.
> Also added an abort on infrastructure failure, after a run of 232 consecutive
> HTTP 401s was recorded as a score of 0.0 rather than as a void run.

The first three were fiscal years attributed from the filing, the grader parsing
the question's year as the answer, and abstentions scored as hallucinations.
The others follow.

A fourth, found 26 Sep and measured without a model call: **the comparative
grader failed correct answers.** "How did X change from FY2023 to FY2024?" never
asks for a percentage; the model answered with both figures and was failed for
"no percentage found". Re-marking `baseline-v5`'s stored answers with the fixed
grader flipped **28 answers from fail to pass and none the other way** —
comparative cases went from 5/48 to 33/48, single-hop was untouched at 124/160.
Every flip is listed in `reports/regrade-baseline-v5-2943b37a.json`, and each
states both XBRL values exactly.

**D12, fixed 5 Oct:** two NVIDIA year-over-year cases compared across the June
2024 10-for-1 split, because each year's value came from its own earliest
filing. The set expected "EPS decreased 75.4%" (11.93 → 2.94) and "shares
increased 893.4%" (2,464M → 24,477M), and the model's split-adjusted answers
were the right ones. Each fact now also stores the prior year as its own filing
printed it (`prior_year_value`, `sql/004`), and comparisons use that: "EPS
increased 147.1%" (1.19 → 2.94) and "shares decreased 0.7%" (24,643M →
24,477M). The other 46 comparative cases already matched their filings and did
not move. The agent's `compare_fact` uses the same basis.

**D13, fixed 5 Oct:** twelve JNJ cases graded fiscal 2023 against fiscal 2022.
JNJ's year ends on the Sunday nearest December 31, so fiscal 2022 ended
2023-01-01. The year-of-period-end rule filed it as 2023, where, as the earlier
filing, it displaced the real fiscal 2023. The six `num-JNJ-*-2023` and six
`yoy-JNJ-*-2023-2024` cases expected fiscal-2022 figures: diluted EPS $6.73 for
$13.72, net income $17.94B for $35.15B, and "total assets decreased 3.9%" where
they rose 7.5%. A year ending in the first week of January now belongs to the
year before, which matches the filings' own `fy` on all eight JNJ year-ends that
fall there. The refresh changed only JNJ's rows: 63 values moved to their
correct year, and 25 missing years were filled in.

**D14, fixed 5 Oct: evidence labels matched on a rounded digit.** The linker
searches chunk text for each case's value, and every needle was the value
rounded to a whole number. Diluted EPS of 2.94 became "3", and the first
needle that hits wins, so EPS cases were labelled with whichever chunks
happened to contain the digit, and hit@k for them measured nothing. Needles now
keep printed decimals ("2.94", "11.80"), and any needle under three digits is
dropped. 38 cases' labels changed, 26 of them EPS. One case lost its labels:
`num-PG-ResearchAndDevelopmentExpense-2024` is exactly $2.00B, its five old
labels came from the needle "2", and no chunk prints the figure as "2,000". It
still counts toward accuracy, but not toward retrieval metrics.

**D15, fixed 5 Oct, after the rebuild: labels out of reach.** The linker
accepted the case's filing or any later one, since later 10-Ks reprint earlier
years, but search only ever looks in the case's own. A fifth of the labels
could never be retrieved and counted as misses in recall@k. Re-linking changed
106 cases' labels and nothing else: the same 232 questions and answers (exam
fingerprint unchanged), 919 references down to exactly the 741 reachable ones,
and no case losing its last label. Retrieval effects are in §1.

**The rebuild, 5 Oct.** It refreshed the facts without re-ingesting filings:
`ingest run` never overwrites an existing fact, and re-parsing a filing deletes
its chunks through `chunks.section_id`, which would move the corpus the set is
graded against. All 232 case ids stayed. Fourteen expected answers changed (the
12 D13 and 2 D12 cases), and so did one question: `nar-UNH-supply-concentration`
picked up the override added on 24 Sep (`ecd0d94`). That matched the
pre-rebuild simulation exactly.

```bash
edgar-intel init --schema sql/004_fact_prior_year_value.sql  # just the new column
edgar-intel ingest facts          # facts only; filings, sections, chunks untouched
edgar-intel ingest verify-facts   # passed: all four checks
edgar-intel eval build --numeric 20 --narrative 3
```

**What the rebuild did to the scores.** It is a new exam, so v6 and v7 do not
compare as a trend. But the same system answered both, under the same config,
so the difference can be attributed case by case:

| | `baseline-v6`, old key | `baseline-v7`, corrected key |
|---|---|---|
| the 14 cases whose expected answer changed | 0/14 | **8/14** |
| the other 194 numeric cases | 154/194 | 154/194 (one flip each way) |
| hit@5 on the 194 cases whose labels did not change | 0.7577 | 0.7577 |
| wrong, non-abstaining numeric answers | 20 | 10, of which 9 fewer are among the 14 |

**Every point of the accuracy difference is the answer key.** Eight answers the
old key failed were right all along: both D12 cases and six of the twelve D13
cases. Re-marking `baseline-v6`'s own stored answers against the corrected key,
with no model call, passes the same eight of the fourteen that v7 passes. They
also account for most of the halved hallucination rate. The six JNJ cases still
failing (2023 gross profit and liabilities, and four 2023→2024 comparisons) are
now real misses, not ground-truth errors.

> Rebuilt the golden set after fixing three ground-truth defects and attributed
> the re-run case by case: the entire 0.038 accuracy difference was eight
> answers the old ground truth had marked wrong, with the 194 unaffected cases
> scoring 154/194 in both runs.

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
> | `baseline-v6-no-rerank-7362c6c7` | 232 | openai | 0.7404 | pre-rebuild set |
> | `baseline-v7-99204312` | 232 | openai | 0.7788 | rebuilt set, 12,000-char context |
> | `exp-ctx20k-834a873e` | 232 | openai | 0.8413 | rebuilt set, 20,000-char context (the §5 experiment) |
> | `baseline-v8-7ae05834` | 232 | openai | 0.8365 | the same config re-run on the D15 labels |
> | `exp-prompt-conventions-c0f6e105` | 232 | openai | **0.8606** | current: plus the answering-prompt conventions (§5) |
>
> The 0.25 is a **20-case smoke run**, not a baseline. And the deeper problem is
> that the defects being fixed *changed the golden set itself* — corrected fiscal
> years, then `PREFERRED_TAG_FAMILIES` changing which XBRL concept a case asks
> about, then the D12–D14 rebuild. **An accuracy delta that straddles a
> ground-truth rebuild compares two different exams.** That includes
> "0.7404 → 0.7788". The defensible claims are the absolute current figure
> (**0.8606** on 232 cases), the defects found, the case-by-case attribution
> above, the context-cap delta in §5, which is single-variable on one exam, and
> the prompt delta in §5, which is single-variable but in-sample.
>
> The one delta that *is* quotable is the re-grade, because nothing else moved:
> same stored answers, same golden set, same tolerance, only the grading code —
> **0.6202 → 0.7548**, with every flip listed. Regenerated answers move a few
> cases between runs even at temperature 0 (one each way between v6 and v7 on
> the unchanged cases), so quote the current reference as the current figure,
> the re-grade as the size of the grader fix, and §5's 0.7788 → 0.8413 as the
> size of the context-cap fix.

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

The same holds for the later runs. `baseline-v6`'s judge reports 0.75,
`baseline-v7`'s 0.79, and `exp-ctx20k`'s, `baseline-v8`'s and the reference's 0.875. Only 12, 14, 4, 4 and 3 of their
answers match a human label (the floor is 20), and no kappa is reported for
any of them. The
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
edgar-intel eval gate --max-regression 0.03                       # against a database run
edgar-intel eval gate --max-regression 0.03 \
    --baseline-file tests/fixtures/ci_baseline.json               # what CI runs
```

**Status: running in CI since 5 Oct, against a committed baseline. It has never
caught a real regression. Do not claim a catch count.**

The `eval-gate` job loads the frozen fixture (two fictional companies, four
10-Ks) and builds the index. It then builds a 24-case numeric golden set from
the fixture's facts, runs the suite with deterministic fake providers, and
compares overall score, numeric accuracy and recall@5 with
`tests/fixtures/ci_baseline.json`. A drop of more than 0.03 on any of them
fails the job. On 24 cases, one changed answer moves accuracy by 0.042, so a
single regressed case is enough to fail it.

It also fails, rather than passing, whenever the comparison would mean nothing:
- the baseline file is missing or malformed;
- the golden set is not the exam the baseline was measured on, which the
  baseline records as a case count and a hash of every question and answer;
- the run graded only part of the exam.

Before this, CI's database started empty, no run labelled "baseline" ever
existed there, and `eval gate` passed with a warning having compared nothing.

**Verified, not caught:** re-running the fixture with `--top-n 1` instead of 8
failed the gate on all three metrics (numeric accuracy 0.1667 → 0.0833, recall@5
0.8333 → 0.6667). That shows the gate works. It is not a regression the gate
found.

**What it does not measure.** The fake model ignores the system prompt and
extracts its answer from the passages it is given. So the gate measures
retrieval, chunking, context assembly and grading. It does not measure how a
real model responds to a prompt change, and its 0.1667 accuracy is not answer
quality. Those numbers come from the paid baseline (§2).

**Regenerate the baseline** only when a change is meant to move these numbers,
and say so in the commit: the diff of the baseline file is the change being
accepted. So far that has happened once, for D15: on the fixture, recall@5
went 0.833 → 1.0 and accuracy did not move. It must come from a fresh database, so that chunk ids match CI's:

```bash
docker run -d --rm --name edgar-ci -e POSTGRES_USER=edgar -e POSTGRES_PASSWORD=edgar \
    -e POSTGRES_DB=edgar -p 5434:5432 pgvector/pgvector:pg16
export EDGAR_DB_DSN=postgresql://edgar:edgar@localhost:5434/edgar \
    EDGAR_LLM_PROVIDER=fake EDGAR_EMBED_PROVIDER=fake
# Run this in a separate worktree, never the main checkout: `eval build` below
# writes evalset/golden.json, which there is the real 232-case golden set.
edgar-intel init
edgar-intel fixture load --path tests/fixtures/mini_corpus.json
edgar-intel index build --strategy section_aware
edgar-intel eval build --narrative 0
edgar-intel eval run --label ci-baseline --git-sha "$(git rev-parse --short HEAD)"
edgar-intel eval save-baseline <run_key printed above>
docker stop edgar-ci
```

**Bullet shape — usable now**

> Wired the evaluation suite into a CI regression gate that blocks a merge on
> any drop above 3 points in overall score, numeric accuracy or recall@5 against
> a committed baseline, and refuses to compare at all when the exam has changed.
> A separate judge-adoption gate vetoes a new judge rubric on a single new false
> negative, regardless of its κ.

The second sentence is the better half, and that gate *has* fired: it blocked
v3_1. The CI gate has not caught anything yet, so say "blocks", never "caught".

---

## 5. Failure attribution

**Produce it**

```bash
edgar-intel eval failures <run_key>
edgar-intel eval context-probe --case-id <case>
```

**Status: measured on the reference run, on `baseline-v8` and `baseline-v7`, and
on `baseline-v5`'s answers before and after the grader fix (D2).**

| | `baseline-v5`, as graded 17 Sep | `baseline-v5`, re-graded 26 Sep | `baseline-v7`, 5 Oct | `baseline-v8`, 6 Oct | **reference, 6 Oct** |
|---|---|---|---|---|---|
| failures | 84 (79 numeric, 5 narrative by the judge) | 56 (51 numeric, 5 narrative) | 51 (46 numeric, 5 narrative) | 37 (34 numeric, 3 narrative) | **32 (29 numeric, 3 narrative)** |
| retrieval misses — nothing labelled in the top 5 | 52 (62%) | 46 (82%) | 43 (84%) | 30 (81%) | **27 (84%)** |
| generation misses — labelled evidence in the top 5, answer still wrong | 32 (38%) | 10 (18%) | 7 (14%) | 6 (16%) | **4 (13%)** |
| unlabelled | 0 | 0 | 1 (the D14 P&G case) | 1 | 1 |

The rule still asks about the top 5, but since the cap fix the model reads all
eight passages, and the labels are a sample. The sharper test reads the context
each answer was given: of the reference run's 29 numeric failures, **24 never had
the expected figure in the model's context, and 5 did.** Most failures are still
evidence that never reached the model.

The grader fix moved this more than any other figure: 22 of the 28 re-graded
answers had been counted as *generation* misses — correct answers the grader could
not read. The other 6 had been counted as *retrieval* misses, correct answers with
nothing labelled in the top 5. That second group shows the rule over-counts
retrieval: the model reads up to 8 passages, and the labels are a sample of the
relevant chunks rather than all of them. Quote the direction, not the decimal.

- `abstention_rate` / `hallucination_rate`: **0.0865 / 0.0529** on the
  reference run. `baseline-v8`, with the earlier prompt, read 0.1058 / 0.0577,
  and `exp-ctx20k` 0.1106 / 0.0481 under the same config as v8, so the
  hallucination difference between those two is two answers. Abstentions fell from
  `baseline-v7`'s 0.1731 because the cap fix delivered evidence the model had
  been correctly saying it lacked. `baseline-v7` read 0.1731 / 0.0481, against
  0.1635 / 0.0962 on `baseline-v6`. The halving is the answer key again.
  Wrong, non-abstaining answers went from 20 to 10, and 9 of those 10 are among
  the 14 corrected cases; on the other 194 it was 10 against 9. `baseline-v5`
  read 0.1587 / 0.2212 as graded and 0.1587 / 0.0865 re-graded, because every
  D2 answer had been sitting in the hallucination bucket. Both defects inflated
  "hallucination" with correct answers.
- `hit@5` 0.7619, so roughly a quarter of cases have no relevant evidence in the
  top 5 — which is where the abstentions live.

**Bullet shape**

> Instrumented per-case retrieval metrics to separate retrieval misses from
> generation misses, and split abstention from hallucination as distinct
> outcomes (0.111 / 0.048). After fixing a grader that had failed 28 correct
> answers and a ground truth that had failed 8 more, most remaining failures
> had no labelled evidence in what the model read, which pointed the work at
> retrieval and context selection rather than the prompt.

It is the harness's own attribution, and it is why the embedder-window question
(§1) had to be settled before the chunking comparison; it now is.

The context probe is the sharper story: for one case all five relevant chunks
sat in the hybrid top-50 at ranks 9, 11, 13, 21 and 29, and `top_n=8` discarded
every one. The run reported a quality failure. It was a truncation failure.

### The context cap: found from six misses, fixed, measured

Six JNJ cases still failed on the corrected answer key. Tracing them showed
that half had retrieval *working*: the evidence ranked 6th or 7th, inside
`top_n=8`, but the 12,000-character cap dropped it before the prompt was
built. section_aware chunks run 2,000–2,350 characters, so the cap admitted a
median of 6 of the 8 selected passages, never all 8. Across the whole set,
the cap cut the only retrieved evidence in 16 cases, and 13 of them failed in
`baseline-v7`. The attribution rule above had counted all 13 as retrieval
misses. Fitting all 8 needs at most 18,075 characters.

`eval run --context-max-chars` was added (`0e96e20`) so the run could use a
different cap and record it truthfully. The config had previously recorded the
constant whatever the run did. Then one paid run changed only the cap:

| | `baseline-v7`, 12,000 | **`exp-ctx20k`, 20,000** |
|---|---|---|
| numeric accuracy | 0.7788 (162/208) | **0.8413 (175/208)** |
| the 13 cases named before the run | 0/13 | **10/13**, none worse |
| every other case | 181/219 | 186/219 (+6, −1) |
| abstention / hallucination | 0.173 / 0.048 | 0.111 / 0.048 |
| hit@5 (retrieval unchanged) | 0.762 | 0.762 |
| mean prompt tokens · cost per run | 3,424 · $0.126 | 4,828 · $0.175 |
| p50 / p95 | 1015 / 1848 ms | 981 / 2441 ms |

The three named cases still failing now have their evidence in the prompt and
are model misses. JNJ's FY2024 revenue sits beside "Worldwide $88,821" and the
model abstained; its FY2025 answer was $199,210M against a "Segments total"
of 94,193; and its cash comparison read a fair-value table's "Cash" row. 20,000
is the shipped default from the commit after `0e96e20`, for the service and
the agent as well as the evaluation.

> Traced six residual failures to a context cap that silently admitted 6 of the
> 8 retrieved passages, and raised it: numeric accuracy rose 0.779 → 0.841 in a
> single-variable run on the same 232-case set, with 10 of the 13 cases
> predicted beforehand fixed and none regressing, for 39% more prompt tokens.

This delta is quotable, unlike the rebuild's: one exam, one index, one knob. A
same-config re-run a day later (`baseline-v8`) scored 0.8365, so the cap is
worth 12–13 cases, and 1–2 cases either way is run-to-run noise.

### Model misses: found by reading the context, fixed in part by the prompt

The labels are a sample of the relevant chunks, so "labelled evidence in the
top 8" cannot say whether the model actually saw the figure. Each stored result
keeps the exact context it was answered from, so `baseline-v8`'s failures were
searched for the expected figure itself. Only its specific printed form counts
(five or more digits, or per-share decimals). The three-digit billions form,
"199" for $199.21B, matched exhibit years and a dividend table. Of 34 numeric
failures, **25 never had the figure in context, and 9 had every needed figure in
front of the model.** Those 9:

| what the model did | cases |
|---|---|
| took the wrong line of the right statement: earnings before noncontrolling interests (P&G ×2, UNH), cash including restricted cash (AAPL), a fair-value table's "Cash" row (JNJ) | 5 |
| took a part for the total: a segment's R&D beside "Total research and development expense" (JNJ); total assets for revenue beside "Segments total" (JNJ) | 2 |
| abstained beside "Worldwide $88,821" (JNJ revenue) | 1 |
| gave "$18.5B → $20.5B" for a year-over-year question, with no percentage, so the rounding implies +10.8% against +10.3% | 1 |

The answering prompt said nothing about reading financial statements. Six
conventions were added (`bee5dc0`): prefer the consolidated statements; take
the total row; net income means the amount attributable to the company; cash
and cash equivalents excludes restricted cash; revenue's usual labels; and a
year-over-year answer gives both figures and the percentage. They are standard
definitions and match the XBRL concepts the set grades against. Then one paid
run changed only the prompt:

| | `baseline-v8` | **`exp-prompt-conventions`** |
|---|---|---|
| numeric accuracy | 0.8365 (174/208) | **0.8606 (179/208)** |
| the 9 targeted cases | 0/9 | 4/9 |
| the other 223 | 195 | 196 (+1, −0) |
| comparative | 38/48 | 42/48 |
| abstention / hallucination | 0.106 / 0.058 | 0.087 / 0.053 |
| cost per run | $0.175 | $0.181 |

**Four were fixed:** the R&D total, both cash comparisons, and the missing
percentage. **Five were not.** The model ignored the net-income rule in all
three noncontrolling-interest cases, still answered total assets for JNJ's
FY2025 revenue, and still abstained beside "Worldwide". Nothing regressed.

**The gain is in-sample.** The rules came from the failures they fix and were
measured on the same set, so the effect on unseen questions is probably smaller.
The absence of any regression among the other 223 is the evidence that they
are general. The prompt is not tuned further against the remaining five, since
that would be fitting the test. For noncontrolling interests, the structural fix
is the agent's XBRL tools, which read `NetIncomeLoss` directly. Measured in §6,
the agent passes all three once its citation bug is fixed. This prompt serves
the evaluation path only; `/ask` runs the agent.

> Separated generation misses from retrieval misses by searching each answer's
> own context for the expected figure: 9 of 34 numeric failures had the evidence
> in front of the model. Adding six standard financial-reading conventions to
> the answering prompt fixed 4 of them with no regressions elsewhere, raising
> numeric accuracy 0.837 → 0.861 (measured in-sample).

---

## 6. Agent reliability

**Produce it**

```bash
edgar-intel eval agent --label agent-<what>   # the golden set through run_agent, graded as eval run grades
edgar-intel agent stats                       # live traffic in agent_traces; eval runs are not written there
```

**Status: measured 6 Oct on all 232 cases, twice: the agent as deployed, and
with a one-line citation fix.** `/ask` serves the agent, so these are the first
accuracy figures for what the deployed service answers. The fixed agent was
then run a second time with nothing changed. Its totals agree within one case,
but 14 cases flip each way between the two runs (below).

**How it is graded.** `eval agent` (`274419a`) calls `run_agent` on each case
and grades the answer with `build_result`, the function `eval run` uses, so
tolerance, parsing and abstention detection are identical. The narrative judge
reads the agent's tool outputs where `eval run` hands it the retrieved passages.
One rule is new: only what `/ask` returns as an answer is graded as one. An
escalated or exhausted run is scored as declined: not passed, counted as an
abstention, never as a hallucination. A confidence-floor escalation's held-back
draft is graded separately, to test the floor, and never counts as a pass. Runs
are recorded with `pipeline: agent`, and `latest_run` filters on it, so
`/stats/eval` and `eval gate` keep reading the retrieve-then-answer reference.
Each case keeps its outcome, steps, tools, the agent/judge cost split and the
trace (`eval_results.agent`, `sql/005`). Traces are not written to
`agent_traces`, which describes traffic.

| run | sha | what |
|---|---|---|
| `agent-pilot-8af9d907` | `274419a` | 20 cases, a seeded draw stratified by kind, $0.014; found the bug and projected $0.15 for the full run |
| `agent-as-deployed-7a4e5c17` | `274419a` | all 232, the agent as `/ask` served it |
| `agent-citation-fix-c3da771e` | `c4a7424` | all 232; differs from the run above only in `get_fact`'s printed citation |
| `agent-citation-fix-dns-retry-387b41f9` | `c4a7424` | the 7 cases a DNS outage errored in the run above, re-run at the same sha |
| `agent-citation-fix-rerun-65631f87` | `6451ac4` | all 232 again, nothing changed, to measure noise. `6451ac4` differs from `c4a7424` only in the void-run detector's regex, tests and docs |

Predictions for both full runs were written before either, informed by the
pilot (`reports/agent-eval-predictions-20261006.json`, 19:45 UTC).

**The bug: 48% of questions escalated over how a citation was spelled.**
`get_fact` printed its source as `XBRL 0000080424-24-000083` and registered the
citation `xbrl:0000080424-24-000083`. The model cited what it read, and the
citation check rejected an id no tool had returned. The model then re-called
`get_fact` to get a citation, loop detection refused the third identical call,
and the agent escalated while holding the right figure. 126 of 232 runs hit the
rejection, and it ended 65 of the 115 that did not answer. It also produced
wrong answers. For UNH's FY2023 net income and revenue, `get_fact` returned the
correct figures; after two refusals the model fell back to text search and
answered the wrong line. The existing end-to-end test missed it because its
scripted model cited the registered form, which its author knew and the model
could not. `c4a7424` prints the id as registered, and a test now pins a model
that copies the id from the tool output.

| | retrieve-then-answer (reference) | agent as deployed | **agent, citation fix** |
|---|---|---|---|
| numeric accuracy | 0.8606 (179/208) | 0.4135 (86/208) | **0.7115 (148/208)** |
| single-hop | 137/160 | 56/160 | 118/160 |
| comparative | 42/48 | 30/48 | 30/48 |
| narrative, by the uncalibrated judge (not usable) | 21/24 | 16/24 | 15/24 |
| escalated | — | **112 (48.3%)** | **52 (22.4%)** |
| not answered (escalated or step ceiling) | — | 115 | 53 |
| numeric answers that were right | 179 of 190 (0.942) | 86 of 98 (0.878) | 148 of 159 (0.931) |
| abstention / hallucination (numeric) | 0.087 / 0.053 | 0.543 / 0.043 | 0.255 / 0.034 |
| the three NCI net-income cases | 0/3 | 2/3 | **3/3** |
| mean model calls per question | 1 | 3.76 | 2.68 |
| mean prompt tokens per question | 4,996 | 2,824 | 2,029 |
| cost per full run (judge included) | $0.181 | $0.137 | $0.098 |
| p50 / p95 | 1230 / 2936 ms | 5304 / 10360 ms | 3632 / 9288 ms |

The fixed run as recorded scores 0.6875 (143/208). Mid-run, seven UNH cases
failed on a DNS lookup error, and an errored case grades as a fail on both
pipelines. The 0.7115 replaces those seven with a re-run at the same sha, in
which 5 passed. Both runs are stored. The outage exposed a harness gap: the
void-run detector knew Linux's DNS wording but not macOS's, so the seven were
never counted as infrastructure failures. `59e0e5b` fixes that, for `eval run`
too.

**What the fix measured: +70 / −9 cases.** 57 of the 70 gains are runs the
citation rejection had ended. 8 of the 9 losses never touched a citation:
different tool paths, two low-confidence escalations, two narrative verdicts.
That is the agent's ordinary churn, measured below at 14 cases each way between
identical runs. The net gain of 62 cases is far outside it, and the fix is
single-variable on one exam. It was found on this set, but it is a bug fix
(printed and registered ids now match), not a rule fitted to cases.

**Run-to-run noise, measured: totals hold, cases do not.** The fixed agent, run
twice with nothing changed:

| | run B (`c3da771e`, with the retry) | run C (`65631f87`) |
|---|---|---|
| numeric accuracy | 0.7115 (148/208) | 0.7067 (147/208) |
| single-hop / comparative | 118 / 30 | 116 / 31 |
| escalated | 52 | 52 |
| wrong numeric answers | 7 | 10 |
| the three NCI cases | 3/3 | 3/3 |
| cost · p50 / p95 | $0.098 · 3.6 / 9.3 s | $0.100 · 3.4 / 7.6 s |

Between the two, 14 cases went from fail to pass and 14 from pass to fail (13
and 14 numeric), and 27 switched between answered and declined.
Retrieve-then-answer moves about one case each way. So for the agent, an
aggregate difference of a couple of cases is noise, as it is for the other
path. A claim that a change fixed particular cases has to beat chance: about
22% of one run's numeric failures pass in the next, and about 10% of its passes
fail.

**Against retrieve-then-answer, the agent loses by 31 cases, and the loss is in
declining, not in being wrong.**
- **When it answers, it is as accurate.** 148 of its 159 numeric answers were
  right (0.931) against 179 of 190 (0.942), and it gave fewer wrong figures
  (7 against 11). It declined 53 numeric questions against 18.
- **It passes 20 of the reference's 29 numeric failures** and declines the
  other 9, answering none of them wrongly. Most were retrieval misses, which a
  fact lookup does not depend on. Flips: +20 / −51 numeric.
- **All three NCI cases pass.** `get_fact` reads `NetIncomeLoss`, the
  attributable figure, which the prompt conventions could not get the
  retrieve-then-answer model to pick (§5). As deployed, the third escalated on
  the citation bug.
- **Comparisons are its weakest kind**: 30/48 in both runs, against 42/48. The
  model guesses tags (`CashAndCashEquivalents` 17 failed calls, `epsDiluted` 12
  in the fixed run), and `compare_fact`'s miss, unlike `get_fact`'s, does not
  list the tags that exist.

**Escalation: the rate, and where it lands.**
- **48.3% as deployed, 22.4% after the fix.** Not the 0% this section warned
  about. But 41 of the 52 remaining escalations are questions the
  retrieve-then-answer path answered correctly. 11 are questions it also
  failed, and escalating those is the right behaviour. As deployed, 93 of 112
  escalations were on questions it answered.
- **The confidence floor is not the cause.** 46 of the 52 are the model calling
  `escalate_to_human`. The other 6 are floor escalations, and all 6 held-back
  drafts were wrong, so the floor kept six wrong answers from users and cost no
  right ones. As deployed: 5 drafts, 1 right.
- **The largest remaining pattern is the calendar.** Non-answers fall on 2 of 32
  FY2023 questions, 12 of 87 FY2024, 28 of 70 FY2025 and 7 of 19 FY2026. 19
  non-answers say the figure is "not available" or not yet filed, 17 of them
  for FY2025–26, sometimes right after `list_coverage` listed that year's 10-K.
  The model's sense of today's date overrides its own tool's output.
- By cause, the fixed run's 53 non-answers are: 27 escalations the model chose
  after coverage or a text search (most of the calendar cases), 14 after a tool
  call came back empty (mostly a fact lookup on a guessed tag), 6 on the
  confidence floor, 4 residual citation rejections (the model still sometimes
  strips the `xbrl:` prefix), one step ceiling and one loop refusal.

**Cost and latency: cheaper, and slower.** $0.098 per full run against $0.181:
2,029 prompt tokens per question against 4,996, because a fact lookup is short
and the agent never reads a 20,000-character context. p50 3.6 s and p95 9.3 s
against 1.2 s and 2.9 s, because it makes 2.7 model calls per question, one after
another.

**Caveats, in order of weight.**
- **The numeric set is generated from `xbrl_facts`, and `get_fact` reads that
  table.** On numeric questions the agent's lookup is close to reading the
  answer key, so its misses are tag choice, year and guardrails, not reading
  comprehension. Retrieve-then-answer has to find the figure in text. The
  comparison describes the two pipelines on this exam. It is not a test of
  reading.
- Case-level churn is high: 14 each way between identical runs. Totals are
  stable to about one case, but any per-case attribution has to clear the 22%
  of failures that pass by chance.
- The narrative judge is uncalibrated (κ 0.42, §3), so none of the 21, 16 or 15
  of 24 is usable.
- Not changed, deliberately: the tag vocabulary, `compare_fact`'s miss message
  and the calendar prior. Each is the next candidate, each needs its own
  single-variable run, and a fix shaped on these cases would be tuned on the
  test.

**Predictions against outcome.** As deployed, 10 of 15 ranges held: escalation
0.483 in [0.45, 0.65], numeric 0.41 in [0.30, 0.45], NCI 2 of 3, cost, steps and
p50. It was wronger than predicted when it answered (hallucination 0.043 above
[0, 0.03]; answers right 0.878 below [0.90, 1.0]), and faster at p95. For the
fix, 3 of 9 held: escalation 0.224, NCI 3/3, steps 2.68. **Numeric accuracy was
predicted at 0.82–0.94, and "more likely than not" to beat retrieve-then-answer.
It came in at 0.71 and lost by 31 cases.** The pilot showed the bug clearly and
hid what was behind it.

**Bullet shape — usable now**

> Evaluated the production tool-calling agent on the same 232-case set and
> grader as the retrieve-then-answer pipeline, the first measurement of what
> the deployed `/ask` serves. Found a citation-format mismatch that escalated
> 48% of questions. The one-line fix cut escalations to 22% and raised numeric
> accuracy 0.41 → 0.71. At 46% lower cost per question than the RAG path, it
> still trailed that path's 0.86, almost entirely by declining (93% of its
> answers were right). The remaining escalations trace to tag-name misses and to
> the model treating fiscal 2025–26 as not yet reported.

**Expect to be asked:** whether 22% is the right escalation rate. It is not.
41 of the 52 escalations were answerable, and the confidence floor, which is the
usual suspect, caused only 6, all correctly. And why the agent's numeric accuracy
is not a fair test of reading filings: it looks the answer up in the table the
answer key came from.

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

With the shipped 20,000-character context, two clean runs measured **p50 981 and
1230 ms, p95 2441 and 2936 ms** end to end (`exp-ctx20k` and the reference;
retrieval about 135 ms of it). Read that as p50 ≈ 1.0–1.2 s and p95 ≈ 2.4–2.9 s
on a laptop. The reference run cost **$0.181 for the full 232-case run** and
**$0.7804 per 1k requests**; the longer prompt adds about 3.5%. `baseline-v8`'s
timings (p50 1188 ms, p95 3455 ms) were taken while a `DELETE` and `VACUUM` ran
on the same database, so they are not quoted.
The 20,000-character context costs 39% more in tokens and about 0.6 s at p95
over `baseline-v7`'s 12,000 ($0.126 per run, $0.5428 per 1k, p95 1848 ms). The
gain bought with it is in §5.

Between `baseline-v6` (p50 892 ms / p95 1732 ms, retrieval 105 ms) and
`baseline-v7` (1015 / 1848 ms, 142 ms), p50 moved about 120 ms with the same
config and no behavioural change on the request path. The only edits there
resolve the `use_rerank` default, and both runs passed it explicitly. Read that
much as run-to-run variance on a laptop, not as a regression. The v6 reranked run: p50
1434 ms, p95 2340 ms, $0.5582 per 1k — the cross-encoder's 542 ms at p50 is
the price that bought nothing (§1).

**Bullet shape — usable now**

> Instrumented per-request token accounting and stage-level timing across
> retrieval, generation and judging: p50 ≈ 1.0–1.2 s / p95 ≈ 2.4–2.9 s end to
> end across clean runs, at $0.18 per full 232-case evaluation and $0.78 per 1k
> requests.

The agent that `/ask` serves is slower and cheaper (§6). After the citation
fix it measured p50 3.6 s and p95 9.3 s at 2.7 model calls per question, and
$0.098 per full run, because its prompts carry fact lookups rather than a
20,000-character context.

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
| numeric accuracy **0.8606** on the current set (retrieve-then-answer path; noise is 1–2 cases) | ✅ measured |
| answering-prompt conventions: 0.837 → 0.861, 4 of 9 model misses fixed, 0 regressions | ✅ measured, single variable, **in-sample** (§5) |
| model misses found by reading each answer's context: 9 of 34 numeric failures had the figure | ✅ measured (§5) |
| context cap 12,000 → 20,000 characters, single variable: **0.7788 → 0.8413**, 10 of 13 predicted cases fixed, none regressed | ✅ measured (§5) |
| the rebuild's +0.038 over v6 is entirely the corrected key: 0/14 → 8/14, the other 194 at 154/194 in both runs | ✅ measured, case by case (§2) |
| "accuracy 0.7404 → 0.7788" | ❌ **not defensible** as a trend — two exams; quote the attribution above instead |
| grader fix re-graded the same answers **0.6202 → 0.7548**, 28 flips listed, 0 reversed | ✅ measured |
| "accuracy 0.25 → 0.62" | ❌ **not defensible** — 20-case smoke vs 232-case baseline, across a ground-truth rebuild |
| eight ground-truth, grading and labelling defects found and fixed (D12–D15 landed 5 Oct) | ✅ measured |
| void-run abort after 232 consecutive 401s scored as 0.0 | ✅ measured |
| hit@5 0.762 / MRR 0.555, and the 0.204 ceiling gap (re-measured on the rebuilt labels) | ✅ measured |
| recall@5 0.408 / nDCG@10 0.429, every label reachable (D15 fixed: was 0.320 / 0.367) | ✅ measured, free sweep |
| reranker off on a single-variable run: +542 ms p50 for no gain | ✅ measured on the v6 set; the free sweep agrees on v7 labels |
| abstention 0.087 vs hallucination 0.053, split | ✅ measured; hallucination's drop from v6's 0.096 is the corrected key, abstention's from v7's 0.173 is the cap fix |
| most remaining failures never had the figure in the model's context (24 of 29 numeric) | ✅ measured on the reference run, by context search; quote the direction |
| corpus: 8 companies, 24 10-K filings, 264 sections, 23,499 chunks, 1,363 XBRL facts | ✅ measured |
| p50 ≈ 1.0–1.2 s / p95 ≈ 2.4–2.9 s, $0.78 per 1k — local, two clean runs | ✅ measured |
| Cohen's κ = 0.42, below the floor, gate suppressed a wrong 79% | ✅ measured |
| judge-adoption gate vetoed a rubric on one false negative | ✅ measured |
| bounded agent: mechanism | ✅ shipped; measured in §6 |
| narrative pass rate | ⚠️ 0.50 by human label on 24 cases; the judge's 0.79 (v5), 0.75 (v6), 0.79 (v7) and 0.875 (exp-ctx20k, v8, current) are not usable |
| two NVIDIA yoy cases compared across a stock split (D12) | ✅ fixed, in the golden set since 5 Oct |
| twelve JNJ cases expected fiscal-2022 figures under a 2023 label (D13) | ✅ fixed, in the golden set since 5 Oct |
| embedder window (D8): a 512 window lifts hybrid hit@5 0.762 → 0.805 but not accuracy (0.841 → 0.837); declined | ✅ measured, single variable (§1) |
| four chunking strategies at an equal 512 ceiling: fixed / recursive / section-aware tie (0.784–0.789), semantic 0.745 | ✅ measured, end to end (§1) |
| chunk size over strategy: 512 → 600 tokens recovers 7 of the 11 cases to the shipped index (0.784 → 0.817 vs 0.837) | ✅ measured (§1) |
| agent (what `/ask` serves), numeric accuracy: **0.4135** as deployed → **0.7115** after a one-line citation fix, against 0.8606 for retrieve-then-answer; 93% of its numeric answers right; 3/3 NCI cases | ✅ measured (§6), same set and grader; fix single-variable but found on this set; a same-config re-run scored 0.7067, so totals hold to ±1 case, while 14 cases flip each way |
| agent escalation rate: **48.3%** as deployed → **22.4%** after the fix; 41 of the 52 remaining escalations were questions RAG answered; the confidence floor caused 6, all with wrong drafts | ✅ measured (§6) |
| "the agent is more accurate than RAG" | ❌ **not true** on this set: −31 cases, and its numeric lookups read the table the answer key came from |
| concurrency / throughput sweep | ❌ needs a deployed instance |
| regression gate in CI | ✅ running since 5 Oct against a committed baseline; fails on a deliberate regression; nothing caught yet (§4) |
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
