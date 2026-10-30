# Model router: plan and predictions

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

## Results

Full run not yet started: it waits for Aditya's go.
