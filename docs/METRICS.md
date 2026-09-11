# From measurement to resume bullet

Every number this repo produces, where it comes from, and the bullet it
supports once **you** have run it and recorded the real figure.

> **The rule that governs this whole document:** run the command, record what it
> actually says, and write the bullet from that. A placeholder left in a resume
> is a claim you cannot defend, and the interviewer will pick exactly the most
> impressive-sounding line to probe first.

---

## 1. Retrieval quality

**Produce it**

```bash
edgar-intel eval compare --strategies fixed,recursive,section_aware,semantic
```

**What comes out**

| run | strategy | overall | numeric | recall@5 | mrr | ndcg@10 | p95_ms | cost_per_1k |
|---|---|---|---|---|---|---|---|---|
| … | fixed | … | … | … | … | … | … | … |
| … | section_aware | … | … | … | … | … | … | … |

**Bullet shape**

> Benchmarked four chunking strategies over a `<N>`-question labelled set on
> `<M>` SEC filings, raising recall@5 from `<baseline>` to `<best>` by moving to
> section-aware chunking with a cross-encoder reranker.

**Where the numbers live:** `eval_runs.summary.retrieval` in Postgres, and
`reports/<run_key>.json`.

**Expect to be asked:** why recall@5 and not accuracy; what recall@k does not
tell you (nothing about ordering — that is what nDCG is for); why reranking can
leave recall unchanged while improving nDCG.

---

## 2. The eval set itself

**Produce it**

```bash
edgar-intel eval build --numeric 20 --narrative 3
```

**Bullet shape**

> Built an evaluation harness of `<N>` cases — `<A>` numeric questions generated
> from SEC XBRL filings data with machine-verifiable answers, `<B>` hand-written
> narrative questions — giving the system a quality baseline that does not
> depend on an LLM grading another LLM.

**Why this one matters most:** evals appear in roughly 68% of AI-company
postings and almost no early-career candidate has built one. This is the highest
signal-to-effort item in the whole repo.

---

## 3. Judge calibration

**Produce it**

```bash
edgar-intel eval run --label baseline
edgar-intel eval label <run_key> --n 40      # ~30 minutes of your time
```

**What comes out:** `kappa=0.74: substantial agreement, narrative scores usable`

**Bullet shape**

> Calibrated an LLM-as-judge against `<N>` human-labelled answers, measuring
> Cohen's κ = `<value>` and gating narrative scores on a κ ≥ 0.60 floor so that
> an unreliable judge fails the build instead of reporting a misleading pass
> rate.

**Expect to be asked:** what κ measures that raw agreement does not; what you
would do if κ came back at 0.3 (tighten the judge rubric, or give up on the
narrative half and say so). Have an answer for the second one — it is the more
interesting question.

---

## 4. Regression gate in CI

**Produce it**

```bash
edgar-intel eval gate --max-regression 0.03
```

**Bullet shape**

> Wired the evaluation suite into CI to block merges on any regression above 3
> points in overall score, numeric accuracy or recall@5, catching `<N>` changes
> that degraded answer quality before they shipped.

Only claim the `<N>` once it has actually caught something. It will — prompt
edits regress quality far more often than people expect. That is the sentence
worth waiting for.

---

## 5. Failure attribution

**Produce it**

```bash
edgar-intel eval failures <run_key>
```

**Bullet shape**

> Instrumented per-case retrieval metrics to separate retrieval misses from
> generation misses, showing `<X>%` of failures were cases where the evidence
> never reached the model — redirecting effort from prompt tuning to chunking.

This is a strong bullet because it demonstrates diagnosis rather than
tinkering, which is the distinction between an engineer and someone following a
tutorial.

---

## 6. Agent reliability

**Produce it**

```bash
edgar-intel agent stats
```

**Bullet shape**

> Engineered a bounded tool-calling agent over 6 typed tools with an 8-step
> ceiling, retry-with-backoff, loop detection and a confidence-floor escalation
> path; across `<N>` runs it answered `<X>%`, escalated `<Y>%` and never
> exceeded its step budget.

**Expect to be asked:** what your escalation rate is and whether it is right. A
0% escalation rate is not a good sign — it usually means the confidence floor is
too low and the agent is answering things it should hand off. Know your number
and have a view on it.

---

## 7. Hallucination prevention

**Bullet shape**

> Added citation and numeric-grounding validation that rejects any answer citing
> a passage no tool returned, or quoting a figure absent from every tool output,
> converting hallucination detection from a model judgement into a code-level
> check.

No number needed. The mechanism is the claim, and it is unusual enough to be
worth a line on its own.

---

## 8. Latency and throughput

**Produce it**

```bash
edgar-intel bench sweep --endpoint /search
```

**Bullet shape**

> Benchmarked the retrieval API across concurrency levels 1–32, holding p95
> latency under `<X>`ms at `<Y>` req/s and identifying saturation at concurrency
> `<Z>` as per-instance capacity.

**Expect to be asked:** why p95 and not mean (a service averaging 400ms with a
p99 of nine seconds is visibly broken for one user in a hundred, and the mean
never tells you); what the knee means.

---

## 9. Cost

Reported on every eval run as `total_cost_usd` and `cost_per_1k_usd`; on every
agent run as `total_cost_usd`.

**Bullet shape**

> Instrumented per-request token accounting across retrieval, generation and
> judging, bringing evaluation-suite cost to $`<X>` per full 200-case run and
> `<Y>`¢ per 1k agent requests.

Cost consciousness is explicitly named in senior AI job descriptions and is
almost absent from early-career resumes. Cheap to add, disproportionately
credible.

---

## 10. Fine-tuning tradeoff

**Produce it**

```bash
edgar-intel finetune dataset
edgar-intel finetune train
edgar-intel finetune benchmark
```

**Bullet shape**

> Fine-tuned `<model>` with LoRA on `<N>` distantly-supervised extraction
> examples, lifting F1 from `<baseline>` (few-shot prompting) to `<result>` on a
> company-disjoint held-out set at `<X>`× lower cost per request, against
> `<H>` GPU-hours of training.

**The comparison is the bullet, not the training.** "I fine-tuned a model and
got 89%" says nothing without the baseline it beat and the cost it traded.

**Expect to be asked:** why you split by company (random splitting inflates the
score via memorised boilerplate); why you included negative examples; whether
the fine-tune was actually worth it. "No, and here is the number that shows it"
is a perfectly good answer — better than a strained yes.

---

## Metrics that carry the most weight, ranked

1. **recall@k with a baseline** — rare on early-career resumes, immediately
   credible
2. **Cohen's κ for judge calibration** — almost nobody has this
3. **p95 latency** — signals you have operated a service, not just built one
4. **$/1k requests** — signals you think about constraints
5. **Eval-set size with provenance** — "200 cases, generated from XBRL"
6. **Regression-gate catches** — signals production discipline
7. **Escalation rate** — signals you understand agent reliability

## Metrics to avoid

- **Accuracy with no baseline.** Unfalsifiable, and reads as such.
- **Mean latency.** Hides the behaviour that matters.
- **"Improved by 20%"** with no starting point. Invites the one question you
  cannot answer.
- **Anything you have not personally measured.** Every number on a resume is a
  promise to explain it under questioning.
