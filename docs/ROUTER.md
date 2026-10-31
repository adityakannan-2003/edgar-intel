# Model router: plan, predictions and results

Written 30 Oct 2026, 19:40 UTC, before the pilot's first call. The
predictions below are scored against the results files afterwards, right or
wrong. The question: which model should answer which question, for the agent
`/ask` serves? Is a cheap model with an escalation path, or a router that
rates difficulty first, worth what it costs against one model for everything?

## What is fixed

- **Exam:** `evalset/golden.json`, 232 questions: 208 numeric graded against
  XBRL (160 single-figure lookups, 48 comparisons) and 24 narrative.
- **Grader:** `grade_agent_run`, the function `eval agent` uses. A run that
  ends without an answer is scored as declined: not passed, never a
  hallucination. The narrative judge is uncalibrated (κ 0.42, METRICS §3), so
  narrative counts are reported but not used. **The accuracy that counts is
  numeric, out of 208.**
- **Agent:** `run_agent` as `/ask` runs it, 8 steps, confidence floor 0.55,
  today's date in the prompt. The only change is the model each run uses.
  Every run records the agent config and index shape in its `.meta.json`.
- **Prices:** list prices from `PRICES_USD_PER_MTOK` (OpenAI's pricing page,
  16 Oct 2026; input / cached input / output per 1M tokens):

  | model | input | cached | output |
  |---|---|---|---|
  | gpt-4o-mini | $0.15 | $0.075 | $0.60 |
  | gpt-6-luna | $0.10 | $0.01 | $0.50 |
  | gpt-6.1-sol | $2.00 | $0.10 | $10.00 |
  | gpt-6-astra | $10.00 | $1.00 | $50.00 |

  Each call is priced from its own reported tokens. Reasoning tokens are
  billed as output.

**Models.** gpt-6-luna has the lowest list price, gpt-6.1-sol is the strong
model, and gpt-4o-mini is what `/ask` runs today. Its last measurement was
200/208 on 6 Oct (`agent-date-in-prompt-a74f3c03`, model cost $0.34 per 1,000
questions). luna's list price is below gpt-4o-mini's, but luna reasons by
default: 234–239 hidden output tokens per call in the fine-tune run. So the
cheaper model per question is something to measure, not assume, and both are
run as the cheap tier. **gpt-6-astra is out.** At 5× sol's prices, one full
run would cost about $6.50, and two would be most of the $10–20 budget for a
single arm.

## Arms

| arm | spec |
|---|---|
| gpt-4o-mini for everything | `single:gpt-4o-mini` |
| luna for everything | `single:gpt-6-luna` |
| sol for everything | `single:gpt-6.1-sol` |
| cascade, mini first | `cascade:gpt-4o-mini,gpt-6.1-sol` |
| cascade, luna first | `cascade:gpt-6-luna,gpt-6.1-sol` |
| difficulty router, mini rates | `difficulty:gpt-4o-mini,gpt-6.1-sol` |
| difficulty router, luna rates | `difficulty:gpt-6-luna,gpt-6.1-sol` |

- **Cascade:** the cheap model runs the agent. If its run ends without an
  answer, sol runs it again from scratch. That covers the model calling
  `escalate_to_human`, a low-confidence answer, a citation or grounding
  rejection past the retry budget, the step ceiling, and no valid JSON. A
  screened refusal is not escalated. Both runs are paid for and timed.
- **Difficulty router:** the cheap model rates the question in one short call
  (`RATE_SYSTEM` in `evals/router.py`). Easy means one reported figure for one
  company and one year; anything else is hard. Easy goes to the cheap model.
  Hard, or a rating that does not parse, goes to sol. The rubric is generic.
  It was not fitted to this set's failures, and it never sees the set's own
  `difficulty` labels. Its agreement with those labels is reported as a check.
- GPT-6 calls go out without `temperature` and `max_tokens`, which those
  models reject, and with `max_completion_tokens` 2048, as in the fine-tune
  benchmark. So they run at temperature 1; gpt-4o-mini runs at 0.

Code: `src/edgar_intel/evals/router.py`, tests in `tests/test_router.py`.

```bash
edgar-intel router run --arm cascade:gpt-4o-mini,gpt-6.1-sol --label router-cascade-mini-r1
edgar-intel router score reports/router/*.jsonl --reference single:gpt-4o-mini --out reports/router/score.json
```

A run appends one graded row per case to `reports/router/<label>.jsonl`. The
same command resumes and retries only the cases that errored. Spend on
superseded rows is still counted.

## Pilot

40 cases, `--sample 40` (a seeded draw stratified by kind and difficulty: 28
lookups, 8 comparisons, 4 narrative), every arm once, at most 4 arm processes
at a time so OpenAI rate limits do not inflate latency. **Then stop and report
to Aditya** with the pilot's cost, what it showed, and the projected cost of
the full run. The full run waits for his go.

**Pilot rules, written now:**
- An arm is dropped from the full run if, after one resume, more than 2 of
  its 40 cases still error, or more than 10% of its agent runs end `failed`
  (no valid JSON).
- If pilot spend passes $2.50, the pilot stops where it is.

## Full run (planned)

Every arm that survives the pilot, run twice on all 232 cases.
- **Accuracy:** numeric correct per run, the mean of the two runs, and a
  bootstrap 95% interval over cases.
- **Noise:** each arm's two runs against each other, as the gap in correct
  cases and as flips each way.
- **When a difference is claimed** (the fine-tune benchmark's rule): only when
  the paired-bootstrap interval against the reference (gpt-4o-mini for
  everything, the deployed agent) excludes zero, *and* the gap in cases is
  bigger than the largest gap between two runs of one arm.
- **Cost:** per 1,000 questions, over all 232. Also per 1,000 correct numeric
  answers: model cost on the numeric questions divided by the numeric ones
  answered correctly. The judge (gpt-4o-mini, on narrative only) is
  reported separately; it is the same for every arm.
- **Latency:** p50 and p95 per question, end to end: the rating call and
  every agent run, tool calls included.
- **Stop rules:** if full-run spend passes $15, the runs stop and the results
  are reported as they stand. An arm with more than 2% of cases still
  erroring after a resume is reported as incomplete.

## Predictions

**Pilot (40 cases):**

1. Every arm finishes all 40 cases with at most one errored case. The GPT-6
   models accept the parameter overrides: no 400s.
2. Pilot spend is between $0.30 and $1.50.
3. Per question, luna costs more than gpt-4o-mini despite its lower list
   price, because of its reasoning tokens. sol costs 8–25× as much as
   gpt-4o-mini.
4. gpt-4o-mini has the lowest p50 latency of the three single-model arms.

**Full run (232 cases, two runs per arm; numeric correct out of 208, per
run):**

5. gpt-4o-mini: 195–203. luna: 188–204. sol: 197–206.
6. Each cascade's mean is 0–5 cases above its cheap arm's. sol answers
   2–8% of questions in the mini cascade and 2–12% in the luna cascade.
7. Each difficulty router sends 25–40% of questions to sol, and its ratings
   agree with the golden set's labels (hard = a comparison or narrative) on at
   least 90% of questions. Its mean is within ±4 cases of its cheap arm's.
   The reason predicted: the cheap model's errors are confident
   single-figure lookups (on 6 Oct, 6 of gpt-4o-mini's 8 numeric failures),
   which the rater calls easy.
8. The two runs of each arm differ by at most 5 correct cases, and no arm
   flips more than 10 cases each way.
9. Against gpt-4o-mini for everything, no arm's accuracy difference is
   claimed under the rule above.
10. Cost per 1,000 questions, mean of the runs: gpt-4o-mini $0.30–0.45;
    luna $0.35–1.20; sol $3.00–9.00; mini cascade $0.35–0.90; luna cascade
    $0.45–1.80; mini-rated router $1.10–3.50; luna-rated router $1.20–4.00.
11. Cost per 1,000 correct numeric answers: gpt-4o-mini for everything is
    the lowest of the seven arms. The mini cascade is within 1.6× of it, and
    sol for everything is at least 10× it.
12. p50 latency per question: gpt-4o-mini 2.0–3.0 s, luna 3.0–9.0 s, sol
    3.0–8.0 s. Each cascade's p50 is within 0.5 s of its cheap arm's, and
    its p95 is higher.
13. Total full-run spend is between $4 and $9.

## Pilot results (30 Oct, 19:35–20:03 UTC)

All numbers come from `reports/router/pilot-*.jsonl` and
`reports/router/pilot-score.json`. The pilot ran at `441f436`, on the same
index as the 6 Oct agent runs (5,468 `section_aware` chunks, ids
57341–62808). 280 cases ran, with no errors and no rejected parameters.
**Spend: $0.67**, judge included. That covers two one-case smoke tests and
the latency re-run below.

| arm | numeric | narrative (judge, not usable) | sent to sol | $ per 1k questions | $ per 1k correct numeric | p50 / p95 |
|---|---|---|---|---|---|---|
| gpt-4o-mini for everything | 35/36 | 2/4 | — | $0.36 | $0.37 | 3.1 / 23.8 s |
| luna for everything | 36/36 | 3/4 | — | $0.38 | $0.24 | 5.1 / 34.1 s |
| sol for everything | 36/36 | 2/4 | — | $5.90 | $3.95 | 4.9 / 28.7 s |
| cascade, mini first | 36/36 | 3/4 | 5% | $0.65 | $0.41 | 3.2 / 22.2 s |
| cascade, luna first | 36/36 | 3/4 | 5% | $1.95 | $0.23 | 5.0 / 44.7 s |
| router, mini rates | 36/36 | 3/4 | 30% | $3.57 | $1.23 | 4.4 / 33.9 s |
| router, luna rates | 36/36 | 3/4 | 30% | $3.26 | $1.19 | 5.9 / 28.5 s |

**What it showed:**
- **Numeric accuracy is at the ceiling for every arm.** gpt-4o-mini's one
  miss is the AAPL revenue comparison it escalated, the same
  `Revenues`-tag miss as on 6 Oct. The mini cascade sent it to sol, which
  answered. On 36 numeric questions nothing separates the arms. The full run
  needs every case, and the gaps may still sit inside the noise
  (prediction 9).
- **Both raters matched the set's own labels on 40 of 40** and sent 30% of
  questions to sol.
- **sol on narrative questions drives the cost.** It runs 4–8 text searches
  per narrative question, at $0.01–0.04 each. The luna cascade's two
  escalations were both narrative and cost $0.068 of its $0.078. The
  difficulty routers cost 9–10× gpt-4o-mini per question, with nothing to
  show for it on these 40.
- **luna reasoned less than in the fine-tune run:** 60 hidden tokens per call,
  not 240. On numeric questions it was cheaper than gpt-4o-mini; overall it
  cost slightly more, because of the narrative questions.
- **The latency is model time.** Tool time has a median of 20–50 ms; each
  process pays one ~12 s embedder load on its first search. Every arm's p95
  is its narrative tail. One gpt-4o-mini call hung for 64 s, a 60 s timeout
  and then a retry.
- **The API was slower today than on 6 Oct.** On the same 40 questions,
  gpt-4o-mini's p50 is 3.1 s against 2.3 s. Re-run alone, it was 3.3 s
  (`pilot-mini-alone`), so our 4 processes are not the cause. Latency is
  therefore compared only within one round of the full run, in which all
  arms run side by side.
- **Found in passing, not fixed here:** OpenAI rejects the agent's action
  schema in strict mode. The 400 takes about 270 ms ("'additionalProperties'
  is required to be supplied and to be false"), then the provider falls back
  to `json_object`. So every agent step `/ask` takes makes one failed
  request first. That is the same for every arm, and it is shared code.

**Pilot predictions scored:**
1. Right: no errors in 280 cases, and no 400s from the GPT-6 overrides.
2. Right: $0.67, inside $0.30–1.50.
3. Right, narrowly on its first half. luna cost $0.38 per 1,000 questions
   against gpt-4o-mini's $0.36, but only because of the narrative
   questions. On numeric questions it was cheaper ($0.24 against $0.37 per
   1,000 correct). sol cost 16× gpt-4o-mini, inside 8–25×.
4. Right: the p50s were gpt-4o-mini 3.1 s, sol 4.9 s and luna 5.1 s.

**Pilot rules:** no arm is dropped (no errors, no `failed` runs).

**Projected full run:** every arm twice on all 232 cases. That is $7.46 in
model cost at the pilot's per-question rates, plus about $0.10 for the judge.
The cascades' and routers' costs hang on a few expensive narrative cases, so
the range is $6–10, under the $15 stop rule. Each round runs all 7 arms side
by side, about an hour per round.

## Results (31 Oct 2026, 14:56–15:48 UTC)

Aditya approved the full run on 31 Oct. All numbers come from
`reports/router/full-*.jsonl` and `reports/router/full-score.json`
(`edgar-intel router score reports/router/full-*.jsonl --reference
single:gpt-4o-mini`). Code: `030c1eb`; index as in the pilot.

- **Rounds:** two, each running all 7 arms side by side, 25–27 minutes per
  round. The prompt date was 2026-10-31 throughout.
- **Errors:** 3,248 cases, 1 error. A GPT-6 call on `nar-UNH-legal` (luna
  cascade, run 1) hit the 2,048-token output cap. It was resumed and
  answered.
- **Spend:** $7.61, judge and the retry included. With the pilot, the
  experiment cost $8.28.

**Numeric accuracy is on the 208 numeric cases, which every run finished.**

| arm | numeric correct, run 1 / 2 | accuracy (95% CI) | vs gpt-4o-mini, cases (95% CI) | wrong · not answered | sent to sol | $ per 1k questions | $ per 1k correct | p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini for everything | 198 / 198 | 0.952 (0.921–0.978) | reference | 7 / 6 · 3 / 4 | — | $0.32 / $0.32 | $0.32 | 2.3 / 2.2 s | 4.3 / 3.8 s |
| **luna for everything** | 204 / 204 | 0.981 (0.962–0.995) | +6.0 (+2.0, +10.5) **claimed** | 4 / 4 · 0 / 0 | — | $0.37 / $0.39 | **$0.24** | 4.1 / 3.5 s | 22.0 / 20.3 s |
| sol for everything | 204 / 204 | 0.981 (0.962–0.995) | +6.0 (+2.0, +10.5) **claimed** | 4 / 4 · 0 / 0 | — | $6.20 / $6.17 | $4.05 | 4.3 / 4.2 s | 21.7 / 23.7 s |
| cascade, mini first | 202 / 203 | 0.974 (0.950–0.993) | +4.5 (+1.5, +8.0) **claimed** | 6 / 5 · 0 / 0 | 2.6% / 3.9% | $0.71 / $1.06 | $0.39 | 2.3 / 2.2 s | 4.2 / 5.0 s |
| cascade, luna first | 204 / 204 | 0.981 (0.962–0.995) | +6.0 (+2.0, +10.5) **claimed** | 4 / 4 · 0 / 0 | 1.3% / 1.3% | $0.75 / $0.95 | $0.25 | 4.1 / 3.5 s | 21.9 / 16.4 s |
| router, mini rates | 202 / 200 | 0.966 (0.940–0.988) | +3.0 (+0.0, +6.5), not claimed | 6 / 7 · 0 / 1 | 31.5% / 31.5% | $3.80 / $3.65 | $1.32 | 2.9 / 2.8 s | 21.3 / 20.8 s |
| router, luna rates | 204 / 204 | 0.981 (0.962–0.995) | +6.0 (+2.0, +11.0) **claimed** | 3 / 4 · 1 / 0 | 31.0% / 31.0% | $3.92 / $3.72 | $1.25 | 5.1 / 4.4 s | 21.7 / 21.9 s |

Column notes:
- "$ per 1k correct" is model cost on the numeric questions divided by the
  numeric questions answered right, pooled over both runs.
- The intervals are a bootstrap over cases. The "vs" column is a paired
  bootstrap against gpt-4o-mini for everything.
- Judge cost (gpt-4o-mini, on narrative only) is $0.003–0.012 per run and
  is not in the cost columns.
- Narrative passes, by the uncalibrated judge and not usable: gpt-4o-mini
  15/17, luna 23/22, sol 19/18, mini cascade 20/20, luna cascade 22/22,
  mini router 19/20, luna router 21/21 (out of 24).

**Noise was small.** The largest gap between an arm's two runs is 2 cases
(the mini-rated router). No arm flipped more than 2 cases each way, and
luna, sol and the luna cascade flipped none, although the GPT-6 models run
at temperature 1. In the agent's 6 Oct version, before the citation, tag and
date fixes, identical runs moved 14 each way; with those fixed, nothing
large is left to churn. So under the rule, any gap above 2 cases with an
interval clear of zero is claimed.

**What separates the arms is six questions, and three mechanisms.**
- **Every GPT-6 run misses the same four numeric questions:** the JNJ
  R&D quirk from METRICS §6. JNJ files $483M, $1.84B and $109M under the
  generic R&D tag, while the answer key uses the "Excluding acquired
  in-process" totals; the fourth miss is the year-on-year change computed
  from them. gpt-4o-mini misses them too. Given this answer key, 204 is the
  ceiling.
- **gpt-4o-mini misses six more in each run**, and luna and sol miss none of
  them:
  - **Tool choice.** For CAT's and UNH's FY2025 revenue, gpt-4o-mini read
    a figure out of filing text (CAT: $72.9B against $67.6B). luna called
    `get_fact`, which returned the reported figure.
  - **No recovery after a tag miss.** On the AAPL and MSFT revenue
    comparisons, `compare_fact` on `Revenues` comes back missing, since
    both file the contract-revenue tag. luna retried with that tag;
    gpt-4o-mini escalated.
  - **The citation format.** On 1–2 net-income lookups per run,
    gpt-4o-mini cited the accession number without the `xbrl:` prefix. The
    citation check rejected it, loop detection refused the third
    identical `get_fact`, and the run escalated holding the right figure.
    These passed on 6 Oct; this is the residual citation problem §6
    mentions.
- **The cascade recovers only what the cheap model gives up on.** The mini
  cascade sent 6 and 9 questions to sol, which answered the AAPL comparison
  and the net-income escalations. CAT and UNH revenue were answered wrongly
  with confidence, so they never reached sol. Hence +4.5 rather than +6.
  luna has nothing numeric left to escalate. Its cascade sent only
  narrative questions and one R&D comparison, for no change in accuracy.
- **The difficulty router helps where the cheap model was not failing.**
  Both raters matched the set's labels on 99.6–100% of questions and sent
  the 72–73 comparisons and narrative questions to sol. gpt-4o-mini
  already got 45 of 48 comparisons. The router fixed 2 of the 3 and left
  the lookup misses, which it rates easy, on gpt-4o-mini. That is +3.0,
  with an interval touching zero.

**Cost per correct answer: luna for everything wins.**
- **luna for everything costs $0.24 per 1,000 correct numeric answers.** The
  luna cascade is $0.25, gpt-4o-mini for everything $0.32, the mini cascade
  $0.39, the routers $1.25–1.32 and sol for everything $4.05.
- **Per question, luna's numeric answers are cheaper than gpt-4o-mini's;
  over all questions it costs more.** It pays a third less per input token
  and reasons only about 130 hidden tokens per question. It costs more per
  1,000 questions ($0.38 against $0.32) because it searches more on
  narrative questions.
- **Every arm that adds sol costs more and gains nothing over luna alone.**
  sol on a narrative question costs $0.03–0.07, which is why one or two
  escalations per run make the luna cascade 2–2.5× luna's bill.

**Latency is what luna costs.**
- Its p50 is 3.5–4.1 s against gpt-4o-mini's 2.2–2.3 s, because each of its
  2.3 calls per question takes longer.
- Every GPT-6 arm's p95 is 16–24 s: a narrative question takes 4–8 text
  searches, each a reasoning call. gpt-4o-mini's p95 is 3.8–4.3 s.
- The mini cascade keeps gpt-4o-mini's latency: p50 2.2–2.3 s, p95 4.2–5.0
  s.

## Predictions scored

Pilot predictions 1–4 were all right (above). Full run, 9 predictions: 4 right (5, 7, 8, 13), 3 partly right (6, 10, 12), 2 wrong (9, 11).

5. **Right.** gpt-4o-mini 198 and 198, inside 195–203; luna 204 and 204, at
   the top of 188–204; sol 204 and 204, inside 197–206.
6. **Half right.** Gains held: the mini cascade +4.5 and the luna cascade
   +0.0, inside 0–5. The mini cascade's 2.6% and 3.9% sent to sol held; the
   luna cascade's 1.3% is below 2–12%.
7. **Right.** 31.0–31.5% were sent to sol; ratings agreed with the labels on
   100% and 99.6% of questions; and both routers stayed within ±4 cases of
   their cheap arm (+3.0 and +0.0). The reason given held for the mini
   router: its misses are lookups rated easy, and its gain is on comparisons
   (45 → 47 of 48).
8. **Right.** The largest gap between runs was 2 cases, and no arm flipped
   more than 2 each way.
9. **Wrong.** Five of six arms are claimed more accurate than gpt-4o-mini
   for everything: luna, sol, the luna cascade and the luna-rated router by
   +6.0, the mini cascade by +4.5. Only the mini-rated router's +3.0 is not
   claimed. The prediction treated 6 Oct's 200 as the ceiling for any
   model, and the GPT-6 models recover from tag misses where gpt-4o-mini
   gives up.
10. **6 of 7 right.** The mini-rated router's $3.72 is above its
    $1.10–3.50. The rest were inside: gpt-4o-mini $0.32, luna $0.38, sol
    $6.18, the mini cascade $0.88 (by $0.02), the luna cascade $0.85, the
    luna-rated router $3.82.
11. **Wrong on its main claim.** luna for everything is the lowest per correct
    answer ($0.24), not gpt-4o-mini ($0.32). The other two parts held: the
    mini cascade is 1.24× gpt-4o-mini and sol is 12.8×.
12. **Half right.** The p50s were inside their ranges (gpt-4o-mini 2.2–2.3
    s, luna 3.5–4.1 s, sol 4.2–4.3 s), and each cascade's p50 was within
    0.1 s of its cheap arm's. "p95 higher" did not hold: the luna cascade's
    p95 was lower than luna's in both runs, and the mini cascade's lower in
    one. The GPT-6 narrative tail moves p95 by more than the cascade does.
13. **Right.** $7.61, inside $4–9.

## Reading

**luna for everything wins on cost per correct answer: $0.24 per 1,000
correct numeric answers.**
- **Against the deployed gpt-4o-mini ($0.32):** luna answers 6 more of the
  208 numeric questions in each run (204 against 198, +2.0 to +10.5 at 95%,
  run-to-run gap 2).
- **Against sol (204, $4.05):** luna ties it at one-seventeenth of the
  price.
- **Why it is cheaper:** its input price is a third below gpt-4o-mini's, and
  its reasoning (about 130 hidden tokens per question) is too short to cost
  that back on lookups.
- **Why it is more accurate:** the extra cases are the agent's own failure
  modes. luna calls `get_fact` instead of reading revenue from text,
  retries with the right tag when one comes back missing, and cites the id
  as printed. On this set, the GPT-6 models already do what a strong model
  in a cascade would.
- **What routing adds on top:** nothing. A cascade only helps when the cheap
  model gives up. gpt-4o-mini's remaining errors are confident ones, and
  luna's only escalations are narrative questions, where sol costs
  $0.03–0.07 each. The difficulty router pays sol's price on the 31% of
  questions the cheap models already answer.
- **What luna costs instead is latency:** p50 1.3–1.8 s slower, and p95
  around 20 s on narrative questions, against about 4 s. If that latency is
  unacceptable, the mini cascade (202–203, $0.39 per 1,000 correct, mini's
  latency) is the one routed arm worth having.

**Caveats, in order of weight:**
- The numeric questions are lookups in the table the answer key was
  generated from (METRICS §6). This measures how reliably each model drives
  the tools, not how well it reads filings.
- The 4 JNJ cases cap every arm at 204.
- The narrative judge is uncalibrated, so narrative quality is not measured,
  but narrative questions drive the GPT-6 arms' cost and tail latency.
- Latency was measured with all 7 arms running side by side, while two
  other sessions used the same API key. gpt-4o-mini's p50 of 2.2–2.3 s
  matches 6 Oct's 2.3 s, so the API was not slow during the run, unlike the
  pilot's day.
- Every arm's latency includes the ~270 ms strict-schema rejection on each
  agent step (pilot, above). A separate session is fixing it.
- gpt-4o-mini scored 198 against 200 on 6 Oct (`b79a49f`). The new misses
  are net-income escalations through the citation check. This is not
  separated from the tool change in `0c7e893` or from noise.
- The router's rubric, the arms and the claim rule were all fixed before
  the first paid call. Nothing was tuned on these results.
