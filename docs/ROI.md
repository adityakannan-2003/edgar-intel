# ROI memo: what edgar-intel costs, what it replaces, when it pays

Written 11 Nov 2026, with no model calls. Every figure is measured (its
results file is named beside it) or a dated price, with two exceptions. The
lookup and check times are **assumptions**, and section 4 tests them at 2×
either way. Question volumes are labelled scenarios.

## Verdict

**edgar-intel pays for its hosting at about 2 questions a month.** As
deployed, an answer costs $0.00032 in model calls and the service costs
$6.70 a month to keep running. If a manual 10-K lookup takes an analyst 5
minutes and checking a cited answer takes 1 (both assumed, not timed), each
question saves about 3.8 analyst-minutes, worth about $3.13. Halving or
doubling either assumption keeps break-even under 20 questions a month. The
answer flips only if checking a cited answer takes about as long as finding
the figure yourself.

So running cost is not what limits this system. Two other things do. It
covers 8 companies' standard line items. Its accuracy is measured on an
exam, not yet on real analysts' questions (section 6).

## 1. What one answer costs

Measured on the same 208 XBRL-graded numeric questions:

| how the question is answered | $ per 1,000 questions | right of 208, run 1 · run 2 | p50 / p95 | source |
|---|---|---|---|---|
| **agent on gpt-4o-mini (what `/ask` serves)** | **$0.32** | 198 · 198 | 2.2–2.3 s / 3.8–4.3 s | `docs/ROUTER.md`; `reports/router/full-score.json` |
| agent on gpt-6-luna | $0.38 | 204 · 204 | 3.5–4.1 s / 20–22 s | same |
| agent on gpt-6.1-sol | $6.18 | 204 · 204 | 4.2–4.3 s / 22–24 s | same |
| retrieve-then-answer, gpt-4o-mini | $0.78 | 179 | 1.0–1.2 s / 2.4–2.9 s | `docs/METRICS.md` §8; `reports/exp-prompt-conventions-c0f6e105.json` |
| retrieve-then-answer, gpt-6-luna | $0.53 | 183 · 184 | 1.9 s / 4.1 s | `docs/LONG_CONTEXT.md`; `reports/long_context/report.json` |
| whole 10-K in context, gpt-6-luna | $1.95 (9.7 questions per filing); $10.71 at one | 206 · 206 as run | 2.6 s / 4.7 s | same, and `cost-by-reuse.json` |

- Agent costs come from eval runs and include a few cents of judge calls,
  which live `/ask` does not make. The one live `/ask` call measured cost
  **$0.000425** and took 4.7 s (`docs/DEPLOY.md`, Cost). That is one call,
  not a distribution.
- After a grader fix, long context re-graded to 207 and 208
  (`reports/long_context/regrade-527e0f0.json`).

## 2. What it costs to run each month

| item | per month | source |
|---|---|---|
| Fly machine, 1 GB shared-cpu-1, always on | $6.70 | `docs/DEPLOY.md`, Cost (Fly's pricing page, 17 Oct 2026) |
| Neon Postgres, free plan | $0 (81 MB of the 1 GB limit) | `docs/DEPLOY.md`; `docs/METRICS.md` §8 |
| model calls | $0.32 per 1,000 questions | section 1 |
| **total at 1,000 questions a month** | **$7.02** | arithmetic |
| total at 10,000 / 100,000 a month | $9.90 / $38.70 | arithmetic |

The question volumes are scenarios, not traffic. At 100,000 a month the
table goes past what has been measured. `/ask` throughput on one machine is
not measured. Only `/search` is, at about 25 requests a second
(`docs/METRICS.md` §8). `/ask` is also rate-limited to 20 a minute per IP.

## 3. What it replaces

The baseline is an analyst looking up a 10-K figure by hand. They find the
filing on EDGAR, open the statement, read the line, confirm the fiscal year
and units, and note the source.

| input | value | kind |
|---|---|---|
| **A.** time for one manual lookup | **5 minutes** | **assumption, not timed** |
| **B.** time to check one cited answer: open the cited fact or passage and confirm it is the right line and year | **1 minute** | **assumption, not timed** |
| analyst pay | $49.79 an hour | documented price: BLS Occupational Outlook Handbook, Financial Analysts, 2025 median pay (page modified 27 Aug 2026, read 11 Nov 2026). This is the wage alone. The employer's full cost is higher, so the dollar savings below are understated. |
| answers right, wrong, declined | 198 · 6.5 · 3.5 of 208 (two-run mean) | measured, `reports/router/full-score.json` |

**What the measured accuracy means for review.** Every answer carries a
citation: an XBRL filing id or a passage that a tool returned. A code check
rejects any citation no tool returned (`docs/METRICS.md` §7). So checking an
answer means opening one source, not searching a filing. 4.8% of questions
come back to the analyst anyway: the 1.7% the agent declines (visible) and
the 3.1% it gets wrong (only a check catches those). Its confidence score
cannot pick out the wrong ones. In the 6 Oct measurement, all 6 wrong
answers came at confidence 0.9–1.0 (`docs/METRICS.md` §6). That leaves two
ways to use it:

- **Check every answer:** 1 minute each, plus a 5-minute manual redo for
  the 4.8%. That is **1.22 analyst-minutes per question instead of 5, or
  76% less time.**
- **Trust answers without checking:** about 0.1 minutes per question (the
  declines only), and about 3 wrong figures in every 100 go out unnoticed.
  That is fine for screening. It is not fine for anything published or
  filed.

A check has to compare the figure against the statement line, not just
confirm the citation exists. The most common wrong answer is a real XBRL
value that J&J files under a misleading R&D tag. All 4 misses in every
GPT-6 run are this one (`docs/ROUTER.md`; `docs/METRICS.md` §6).

## 4. Break-even, and what happens if the assumptions are wrong

With checking, each question saves 3.78 analyst-minutes, worth $3.13 at
$49.79 an hour, against $0.00032 in model calls. The $6.70 month is paid
off at **2.1 questions**. Every question after that is a net saving. At
1,000 questions a month (scenario), the system costs $7.02 and saves about
63 analyst-hours (about $3,134).

Break-even in questions a month, and the analyst time saved, with each
assumption halved and doubled:

| | check takes 0.5 min | **check takes 1 min** | check takes 2 min |
|---|---|---|---|
| lookup takes 2.5 min | 4.3 · 76% saved | 5.8 · 56% | 19.5 · 17% |
| **lookup takes 5 min** | 1.9 · 85% | **2.1 · 76%** | 2.9 · 56% |
| lookup takes 10 min | 0.9 · 90% | 0.9 · 85% | 1.1 · 76% |

- **Off by 2× either way, the verdict holds.** Break-even stays between 1
  and 20 questions a month in every cell.
- **The weak corner is quick lookups with slow checks.** Lookups of 2.5
  minutes and checks of 2 minutes still save 17%, but that is a thin case.
- **What would flip it:** a check taking about 97% as long as a lookup
  (4.8 of 5 minutes). At that point checking a cited answer is no faster
  than finding the figure yourself.

## 5. Choices that would move the numbers

- **Switch `/ask` to gpt-6-luna.** It costs $0.06 more per 1,000 questions
  ($0.24 against $0.32 per 1,000 *correct* answers). It sends 1.9% of
  questions back to the analyst instead of 4.8% (4 wrong, 0 declined;
  `docs/ROUTER.md`). That saves 0.12 analyst-minutes per question, about
  $0.10, against $0.00006 more in model calls. The cost is latency: p95
  goes from about 4 s to about 20 s, because narrative questions take 4–8
  searches. One
  analyst-minute buys about 2,600 answers at the deployed price, so
  accuracy is the lever, not the API price. **Recommendation:** switch, if
  a 20 s worst case is acceptable.
- **Whole-document long context** beat retrieval on the same model, 0.99
  against 0.88 (`docs/LONG_CONTEXT.md`). Even at its worst measured reuse, $10.71
  per 1,000, an answer costs about a cent, one 77th of an analyst-minute.
  The architecture choice does not move the ROI.
- **Self-hosting a fine-tune does not pay at this scale.** This was
  measured on project #2's extraction task, a different workload
  (`docs/FINETUNE_RUN.md`, `docs/SERVING.md`;
  `reports/serving/breakeven.json`). An always-on RTX 4090 costs $649.70 a
  month. It beats gpt-6.1-sol above 314k–421k requests a month, gpt-4o-mini
  above 1.33M–2.24M (priced, not run), and gpt-6-luna above 3.02M. At 1,000
  questions a month the rent would be 93× this system's whole running cost.
  The fine-tune is also less accurate: 0.91 F1 against luna's 0.97 and
  sol's 0.98.

## 6. What this memo does not show

- **The lookup and check times are assumptions.** Timing one analyst on
  about 20 real lookups, by hand and then checking the agent's answers,
  would replace them. That is about an hour of an analyst's time.
- **Accuracy is from an exam.** The numeric questions were generated from
  the XBRL table that the agent's `get_fact` tool reads, so on those
  questions the agent is close to looking up the answer key
  (`docs/METRICS.md` §6). **Accuracy on real analysts' questions is not
  measured.**
- **Scope.** It covers 8 companies, 24 10-Ks, 1,363 XBRL facts and 13
  tags (`docs/METRICS.md`, Scoreboard and §6). How it handles questions
  outside that scope is not measured here.
- **Narrative questions are not costed.** The judge that grades them is
  uncalibrated (κ 0.42). By human label, the retrieve-then-answer path
  passes 0.50 of 24 (`docs/METRICS.md` §3 and Scoreboard). This memo makes
  no claim for them.
- **Not counted:** engineering time, and the paid experiments behind these
  numbers. The four write-ups cited here cost $17.12: routing $8.28
  (`docs/METRICS.md` §6), long context $2.12 (`docs/LONG_CONTEXT.md`),
  fine-tune $5.87 (`docs/FINETUNE_RUN.md`) and serving $0.85
  (`docs/SERVING.md`). Earlier eval runs are not included.
- Neon's free plan also limits compute. How close the always-on service
  runs to that limit is not checked.
