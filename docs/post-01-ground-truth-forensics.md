# My evaluation harness was wrong, and it took four bugs to notice

*Draft 1 — 24 Sep 2026. Numbers below are traced to report files in the repo; the
one section that still needs a re-check is flagged at the end.*

I built a retrieval and evaluation system over SEC 10-K filings. The pitch to
myself was that ground truth would be the easy part: XBRL company facts come
from the filer, so a question like *"what was Caterpillar's R&D expense in
FY2025"* has an answer nobody has to argue about. Generate a couple of hundred
of those, and you have a real eval set in an afternoon instead of the week it
takes to hand-write one.

That was right about the afternoon and wrong about everything else. The first
four things I found were all defects in the measuring instrument, and each one
had been quietly changing the score.

## 1. The fiscal year on the fact was not the fact's fiscal year

The SEC's `companyfacts` API gives you every XBRL fact a company has ever
reported, each stamped with `fy` and `fp`. It is extremely easy to read those as
"the period this fact describes." They are not. They identify **the filing the
fact appeared in.**

A 10-K carries three years of income statement. So Caterpillar's FY2025 10-K
emits FY2025, FY2024 *and* FY2023 revenue — and all three arrive labelled
`fy: 2025`.

My dedupe key included `period_end`, so all three survived, under one fiscal
year, and nothing raised. The golden set then contradicted itself in a way that
is obvious once you look and invisible if you do not:

- `num-CAT-Revenues-2023` and `num-CAT-Revenues-2024` carried the **identical**
  expected value.
- `num-CAT-ResearchAndDevelopmentExpense-2025` appeared **twice**, expecting
  2,148,000,000 in one case and 2,108,000,000 in the other.

The system was being graded against a set that disagreed with itself. Every
answer it gave to one of those pairs was wrong by construction.

The fix is to derive the fiscal year from the period the fact actually covers,
which sounds trivial and is not:

- **Duration facts** (revenue, expenses) must span a full year. But "a full
  year" is not 365 days: Apple and Costco are 52/53-week filers, and leap years
  exist. The band that works is **340–380 days**. Tighter and you drop real
  full-year facts; looser and you start admitting three-quarter stubs.
- **Instantaneous facts** (cash, total assets) have no duration at all. You
  cannot tell a fiscal-year-end balance from a quarter-end balance by looking at
  the fact. So they are kept only at dates *proven* to be fiscal year-ends by the
  same company's own duration facts.
- When two filings report the same period, **the earliest accession wins**. That
  is deliberate: "FY2023 revenue" should resolve to what the FY2023 10-K said,
  not to a later restatement, because the retrieval corpus *is* that filing's
  text. Grading against a restated figure asks the system to find something that
  is not in the document.

That last one is the decision I would most want to be asked about. It is not
obviously correct — it is correct *given what the system is allowed to read.*

## 2. The grader read the question back to itself

> *"R&D expense for Caterpillar in FY2025 was $2,148 million"*

The number extractor returned **2025**.

It was scanning for the first numeric token and the question's own fiscal year
was sitting right there in the answer's first clause. The fix is unglamorous —
strip year labels, discard bare four-digit years, prefer candidates carrying
numeric signal like a currency symbol, a scale word, or a decimal — and the
lesson is that a grader is a parser, and parsers need tests with adversarial
inputs. Mine had tests with polite inputs.

## 3. "I don't know" was being scored as a lie

Every failure was one bucket. So an answer of *"the filings provided do not
contain this figure"* was counted identically to an answer that confidently
stated a number that appears nowhere.

Those are opposite behaviours. One is the system working correctly at the edge of
its evidence; the other is the failure mode the entire project exists to prevent.
Collapsing them means you cannot tell whether a change made the system more
honest or more reckless. Split apart on the current run:

| | |
|---|---|
| abstention rate | **0.1587** |
| hallucination rate | **0.2212** |

The abstentions turned out to be the single largest failure bucket, which
immediately reframed the work: they are not a generation problem, they are
retrieval not putting the evidence in front of the model.

## 4. A run that failed 232 times recorded a score of zero

`reports/baseline-45536dc9.json`. 232 cases. `401 Unauthorized` on every single
one. `overall_score: 0.0`.

Zero is a *number*. It goes in the table, it plots, it looks like a data point.
It sits next to the runs that actually ran and drags every average down. A run
that could not reach the model has no score at all, and the difference between
"scored zero" and "produced no score" matters more than almost anything else in
a results table.

Runs now abort once infrastructure errors pass 25% of cases. Two other reports in
the directory also carry 0.0 for the same reason, and one more carries 0.0337 —
which is worse, because that one looks plausible. It was run against the **fake**
provider. The config block says so. I did not read the config block.

**So: read the config block of every report before you read its numbers.** Four
runs in this project carried a provider, a setting, or a git SHA that did not
match what I thought I was running.

## The one that actually scared me: a reporting bug hiding a system bug

Narrative questions — the hand-written ones graded by an LLM judge — scored
**0/24 in every single run** for a week.

And no narrative failure appeared in any report. Not one, across every report
file. The reason is four characters long. The failure-sample code was:

```python
failures = [r for r in results if not r.passed][:50]
```

Numeric cases come first in result order. There were more than fifty numeric
failures. So all twenty-four narrative failures were truncated away, every time.

A reporting defect was concealing a system defect. The metric said "narrative
quality is zero" and the evidence that would have explained it was being silently
dropped by the thing whose job was to show it to me. Failures are now sampled
**per kind**, 25 of each, so no category can be crowded out by a noisier one.

The underlying cause, once visible, was the judge: a model that reasoned before
answering, a 300-token budget, and a JSON parse that threw and was caught into
`verdict=False`. Zero out of twenty-four, no visible error.

## The bit I got wrong twice: my controls were flattering me

To test whether the judge could catch a bad answer, I generated mutations of good
answers — contradictions, omissions, hedges, fabrications — and checked the judge
caught them. Omission controls scored **8/8**.

In the same period, on the same judge, real omissions slipped past **seven
times**.

The controls were worthless and I trusted them. Truncating an answer to simulate
an omission produces something obviously stunted, and any judge catches that. A
*real* omission is fluent, confident, well-written, and covers one part of a
multi-part disclosure. Nothing about it looks wrong. My mutations tested a
failure mode that does not occur.

Worse: one of my "clean controls" report files scored 1.000 on a set that
contained **no omission control at all**. A perfect score on a test with the
question removed.

What replaced it: 24 human labels. The judge agreed with me on 17 of 24 — 70.8%,
which reads fine — and Cohen's κ came back at **0.42**, which says roughly half
that agreement was chance. All seven errors were in the same direction: **7 false
positives, 0 false negatives.** A permissive judge, not a noisy one. So the fix
is a rule the rubric is missing, not a bigger model.

I wrote three more rubrics. In the configuration the system actually runs in, v2
and v3_1 produce **byte-identical** confusion matrices. The rubric stopped
mattering. And the one variation that did move the numbers moved them the wrong
way: giving the judge the source passages to check against *lowered* κ from 0.58
to 0.42, because with evidence in hand it stopped failing anything at all.

The κ ≥ 0.60 floor I had built earned its keep here. The run reports a narrative
pass rate of **0.7917**. The human labels say **0.50**. The gate is why the 0.79
is not on my resume.

## The number I cannot quote, and why that is the interesting part

The obvious way to write this up is "fixed the eval harness, accuracy went from
0.25 to 0.62."

I cannot say that. The 0.25 is from a **20-case smoke run**; the 0.62 is from a
232-case baseline. Different set, different size. But the real problem is deeper
than the sample size: **fixing the ground truth changed the exam.** Corrected
fiscal years changed which values cases expected. A later change to XBRL tag
selection changed which concept some cases even asked about. An accuracy number
measured before and after that is two different tests, and the delta between them
means nothing.

What I can defend is the absolute figure — **0.6202 numeric accuracy on the
current 232-case set** — and the list of defects found. Which is a less
impressive-sounding sentence and a much better one to be asked about.

## The finding I did not expect: verifiable does not mean unambiguous

This is the part I would keep if I had to throw the rest away.

After the harness was fixed, a stubborn set of numeric failures remained. I went
through them expecting more bugs. Instead, straight from the report:

| case | expected | the model answered |
|---|---|---|
| `num-JNJ-ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost-2023` | 14,603M | 15,085M |
| `num-PG-NetIncomeLoss-2025` | 15,974M | 16,065M |
| `num-PG-NetIncomeLoss-2024` | 14,879M | 14,974M |

The first is R&D *excluding* acquired in-process R&D against the figure
*including* it. The second and third are net income *attributable to the parent*
against the figure *including non-controlling interests*. On FY2025 those two are
**0.5697% apart against a tolerance of 0.5%** — it failed by seven hundredths of
a percentage point of definitional difference.

**Every one of those "wrong" answers is a real figure, printed in the filing, in
a table the model correctly read.** My golden set picked one XBRL concept. The
document contains a closely related one, and the question — *"what was net
income"* — does not distinguish them. The model is not hallucinating. It is
answering a question I asked badly.

And the error does not stay put. The derived year-over-year case inherits it:

| case | expected | the model answered |
|---|---|---|
| `yoy-PG-NetIncomeLoss-2024-2025` | increased 7.4% | *"from $14.879 billion to $16.065 billion … approximately 8%"* |

The model's arithmetic is correct. It used my FY2024 figure and the
including-NCI FY2025 figure, so one concept mismatch in one year corrupts a
percentage in a different case. P&G net income appears in three cases in this set
and **all three fail from the same single decision about which XBRL tag to ask
about.** That is the part I had not appreciated: these failures are not
independent. A handful of tag choices is generating a large share of the 79
numeric failures, so the remaining headroom is much smaller and much more
tractable than "62% accuracy" suggests.

The tempting fix is to widen the tolerance until they pass. That hides it. The
honest options are to name the concept in the question, or to record a set of
acceptable concepts per case with the reason each is acceptable.

## And one case where the question is simply wrong

Separate from the ambiguity, the clearest single case in the whole report:

| case | expected | the model answered | why it failed |
|---|---|---|---|
| `yoy-CAT-ResearchAndDevelopmentExpense-2023-2024` | decreased 0.0% | *"FY2024 was $2,107 million, a decrease from $2,108 million in FY2023"* | no percentage found in the answer |

Caterpillar's R&D went from $2,108M to $2,107M. That is a 0.05% change, which my
own expected answer renders as **"decreased 0.0%"**. The model gave both figures
and the direction — it is not merely close, it is *exactly right* — and the
grader failed it for not saying a number that rounds to zero.

The grader is working correctly. The question is wrong. *"How did R&D change from
FY2023 to FY2024?"* is answered by two absolute figures and a direction, and a
grader that demands a percentage is testing compliance with a format I never
asked for. Rewording the question to demand a percentage would make the eval
easier rather than more honest; the fix is to accept either a percentage within
tolerance **or** two absolutes matching the two years, and derive the change.

I came in believing that machine-verifiable ground truth eliminated the
ambiguity problem. It does not. It eliminates *arithmetic* disagreement and
leaves *semantic* disagreement completely intact — and semantic disagreement is
the harder half. The lesson generalises past finance: if your eval set was
generated from a structured source, it inherited that source's choices about what
each field means, and your questions have to carry those choices explicitly or
your model will be punished for reading the document properly.

## What I would tell someone starting one of these

- **Your eval set is code and it has bugs.** Mine had four before it had a
  single useful measurement.
- **Read the config block of every report before the numbers.** Every time.
- **Make a void run produce no score rather than a zero.**
- **Sample failures per category.** A truncated failure list will hide an entire
  broken subsystem, and it will do so silently.
- **Synthetic controls test the failure mode you imagined.** Get human labels.
  Mine disagreed with my judge on 7 of 24 cases, all in one direction, and no
  mutation I invented had found any of them.
- **An accuracy delta that straddles a ground-truth change is not a delta.**

The part nobody warns you about is that all of this is *invisible while it is
happening*. Every one of these bugs produced a plausible number. Nothing crashed.
The system reported 25% accuracy and a narrative score of zero and an agreement
rate of 70.8%, and all three were artefacts of the instrument. If a measurement
looks disappointing, the most likely explanation is not that your system is bad.
It is that you are measuring the wrong thing, in a way that will keep looking
reasonable until you go and check.

---

### Provenance

| claim | source |
|---|---|
| numeric accuracy 0.6202, abstention 0.1587, hallucination 0.2212, narrative 0.7917, p50 996 ms / p95 1700 ms, $0.5435 per 1k | `reports/baseline-v5-2943b37a.json` (`git_sha a9e9c08`) |
| κ 0.4167, 17/24 agreement, FP 7 / FN 0, v2 ≡ v3_1 with source context | `reports/judge_kappa_context_v2_vs_v31.json` |
| κ 0.5833 context-free, v3 at 0.6667 | `reports/judge_kappa_v2_v3_v31.json` |
| 232 cases / 401s / `overall_score: 0.0` | `reports/baseline-45536dc9.json` |
| the fake-provider run at 0.0337 | `reports/baseline-c8d68dd6.json` |
| the 20-case 0.25 | `reports/smoke20-011be57c.json` |
| omission controls 8/8 | `reports/judge_v2_with_omission.json` |
| the clean-controls set scoring 1.000 with no omission control | `reports/judge_v2_clean_controls8.json` |

### ⚠️ Before publishing

- ✅ **Every figure in the two tables above is now read from
  `reports/baseline-v5-2943b37a.json`** (24 Sep) — case ids, expected values,
  answers and failure reasons verbatim.
- ❌ **Two examples were cut** rather than published unverified: JNJ net income
  continuing-ops vs total (17,941 / 35,153) and JNJ diluted EPS (6.73 / 13.72).
  Both were carried from the evidence log and **neither appears in the report**,
  because the failure sample is capped at 25 per kind and 54 of the 79 numeric
  failures are not in the file. Restore them only from a database query, not from
  the log.
- Confirm the CAT R&D pair (2,148,000,000 / 2,108,000,000) — the duplicate
  expected values from the *pre-fix* set. The current set has FY2023 R&D at
  2,108M, consistent with the story, but the 2,148M half is a historical artefact
  and is not in any current report.
- Confirm the **1.000** on the control set with no omission control, and the
  **8/8** omission result, by opening `reports/judge_v2_clean_controls8.json` and
  `reports/judge_v2_with_omission.json`. Both are carried from the evidence log.
- Decide whether to name the repo and link it. If yes, it needs the live URL
  first — see `docs/DEPLOY.md`.
