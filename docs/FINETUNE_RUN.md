# Fine-tune benchmark: full-run plan and predictions

Written 17 Oct 2026, before the full run's first call. Aditya approved it
with a $7 cap. The predictions below are scored against the run's results
file afterwards, right or wrong. The pilot that led here, and the three
dataset changes it forced, are in `FINETUNE_PILOT.md`.

## What is fixed

- **Held-out set:** v3, 532 cases (427 with facts, 105 without, 1,734
  labels) from 10 companies, one per sector. `data/finetune/held_out.jsonl`,
  sha256 `e8e92b026c3c…`, built at `627b616`.
- **Training set:** v3, 1,743 examples from 30 other companies.
  `data/finetune/train.jsonl`, sha256 `d31974d753d3…`.
- **Few-shot examples** for the prompted arms come from training companies
  only: CI, HON, DE, and a CAT table with no facts.
- **Inputs:** every arm gets the same instruction and the same user message
  per case.
- **Scoring:** micro F1 over (concept, year, value), the scorer at
  `8a5275b`. Facts that are dropped or unverifiable are set aside for every
  arm.

## Arms

| arm | runs | where |
|---|---|---|
| gpt-6.1-sol few-shot | 2 passes | OpenAI API, default temperature (1) |
| gpt-6-luna few-shot | 2 passes | OpenAI API, default temperature (1) |
| Qwen3-8B + LoRA | 2 training seeds (17, 29), 2 epochs, all 1,743 examples | vLLM on one RTX 4090 |
| Qwen3-8B few-shot, no LoRA | 1 pass | same server |

gpt-6-astra is out, under the pilot's written rule.

**Hardware:** RunPod Secure Cloud, RTX 4090, $0.89/hr. At 18:38 UTC the
catalog read showed no Community Cloud 4090s at any CUDA version.

**Latency and cost:**
- Each Qwen arm runs one request at a time, so its latency is comparable
  with the API arms'.
- At-load throughput for the seed-17 adapter is measured separately: 200
  requests with 32 in flight, answers discarded.
- The API arms run as four processes side by side, because OpenAI's
  latency does not depend on our own concurrency.

## How results are read

- **The headline:** seed 17 against sol's first pass, on micro F1 with a
  paired-bootstrap 95% interval, plus cost per 1,000 requests (one at a time
  and at load), p50/p95 latency, and training GPU-hours.
- **Noise:** sol pass 1 against pass 2, luna pass 1 against pass 2, and
  seed 17 against seed 29.
- **When a difference is claimed:** only when its interval excludes zero
  *and* it is larger than the biggest same-configuration spread among those
  three.
- **Failed calls** are retried by resuming and never scored as an empty
  answer. If more than 2% of an arm's cases still fail, the arm is reported
  as incomplete.

## Stop rules

- If API spend passes $3.50, the API arms stop and the run is reported as
  it stands.
- If the pod has run for 4 hours, it stops.
- The pod is stopped when the run finishes, and a fresh read confirms it.

## Predictions

1. sol micro F1 is between 0.970 and 0.995; luna is between 0.955 and
   0.990.
2. Each API arm's two passes differ by 0.010 or less, and flip at most 10
   of 532 cases.
3. The LoRA seeds score between 0.85 and 0.95 micro F1, and differ from
   each other by 0.03 or less.
4. sol beats the LoRA on F1 with an interval that excludes zero.
5. Untuned Qwen3-8B few-shot scores between 0.10 and 0.30.
6. The LoRA's most common error is `missed`; untuned Qwen's is
   `wrong_value`.
7. Training takes 0.6–0.8 GPU-hours per seed.
8. The LoRA costs $0.02–0.06 per 1,000 at load and $0.30–0.70 one at a time
   on the $0.89/hr pod. Both are below sol's ~$2.30. At load it is below
   luna's ~$0.20; one at a time it is above it.
9. The LoRA's p50 latency is below both API arms'.
10. Total spend is between $4 and $6.

## Results (17 Oct 2026)

All numbers come from `reports/finetune/full/report.json` (every arm against
sol's first pass) and the two noise files beside it. The set is the 532 v3
held-out cases, every arm finished all of them, and no failed call was left
unretried.

| arm | micro F1 | precision / recall | facts on the 105 no-fact passages | $ per 1k, one at a time | $ per 1k at load | p50 / p95 |
|---|---|---|---|---|---|---|
| gpt-6.1-sol few-shot, pass 1 | **0.980** | 0.983 / 0.977 | 0 | $2.07 | n/a | 2.2 s / 5.5 s |
| gpt-6.1-sol few-shot, pass 2 | 0.980 | 0.983 / 0.977 | 0 | $1.54 | n/a | 2.1 s / 5.7 s |
| gpt-6-luna few-shot, pass 1 | 0.973 | 0.976 / 0.969 | 0 | $0.22 | n/a | 2.5 s / 11.1 s |
| gpt-6-luna few-shot, pass 2 | 0.966 | 0.970 / 0.961 | 0 | $0.22 | n/a | 2.4 s / 10.2 s |
| Qwen3-8B + LoRA, seed 17 | **0.908** | 0.897 / 0.920 | 2 | $0.58 | **$0.061** | 1.9 s / 7.1 s |
| Qwen3-8B + LoRA, seed 29 | 0.912 | 0.906 / 0.918 | 0 | $0.58 | not measured | 1.9 s / 7.1 s |
| Qwen3-8B few-shot, no LoRA | 0.439 | 0.402 / 0.483 | 3 | $0.68 | not measured | 2.1 s / 7.0 s |

**How the costs are computed:**
- **API arms:** reported tokens at list price (sol's two passes differ by
  how much of the prompt was cached).
- **Qwen, one at a time:** the pod's $0.89/hr times mean latency.
- **Qwen at load:** $0.89/hr ÷ 4.07 requests/s, the rate measured with 32 in
  flight over 200 requests (`throughput-lora-s17-c32-n200.json`). On a
  $0.34/hr Community 4090 the same arithmetic gives $0.023, but none was
  available, so it was never measured.

**Same-configuration noise:**

| pair | micro F1 difference | 95% interval | cases flipped |
|---|---|---|---|
| sol pass 2 − pass 1 | +0.0002 | [−0.004, +0.004] | 5 |
| luna pass 2 − pass 1 | −0.007 | [−0.017, +0.001] | 29 |
| LoRA seed 29 − seed 17 | +0.004 | [−0.008, +0.015] | 41 |

The largest same-configuration spread is 0.007 (luna). Under the rule, a
difference is claimed only when its interval excludes zero and it is bigger
than that.

**Against sol pass 1:**

| arm | difference | 95% interval | read |
|---|---|---|---|
| LoRA seed 17 | −0.072 | [−0.089, −0.056] | **sol is better** |
| LoRA seed 29 | −0.068 | [−0.084, −0.053] | **sol is better** |
| luna pass 1 | −0.007 | [−0.013, −0.002] | at the noise floor: not claimed |
| luna pass 2 | −0.014 | [−0.024, −0.005] | sol is better |
| untuned Qwen | −0.541 | [−0.581, −0.499] | sol is better |

What the LoRA bought over its own base: 0.439 → 0.908–0.912, both seeds.

**Training** (`reports/finetune/full/gpu/s17`, `s29`):

| | seed 17 | seed 29 |
|---|---|---|
| data | 1,743 examples, 2 epochs, 1.87M tokens an epoch | same |
| trainer runtime | 2,753.9 s, **0.765 GPU-hours** | 2,748.1 s, **0.763 GPU-hours** |
| speed | 1,362 tokens/s | 1,364 tokens/s |
| loss | 0.037 | 0.040 |

**Where each arm breaks** (per-concept F1 in `report.json`):

- **sol** fails mostly by omission: 40 `missed`, 17 `unlabelled`, 8
  `wrong_value`, 2 `wrong_year`, 3 `not_in_passage`. Its weakest concept is
  common shares outstanding (0.903); its weakest company is UNH (0.942).
  33 of these 70 items are label artifacts, including all 8 wrong values
  and all 3 `not_in_passage` (see "Why the fine-tune loses" below).
- **The LoRA** misses more (139 `missed`) and reports figures that are not
  labels (101 `unlabelled`). It reports 16 `not_in_passage`, and facts on 2
  of the 105 no-fact passages, which no API arm did.
  - 13 of the 16 are figures the passage does not print. The other 3 are
    Nvidia's restated EPS, which is printed; sol reports it too.
  - The 2 no-fact facts are misread printed lines, not inventions.
  - Weakest concepts: shares outstanding (0.760), cash (0.841), equity
    (0.855).
  - Weakest companies: UNH (0.850) and COST (0.869).
  - EPS, where the answer is the printed number, is its best concept
    (0.970).
- **Untuned Qwen3-8B** emits 217 unusable facts and 675 wrong values, mostly
  the scale error the pilot found ("in millions" written with one or two
  zeros too many), and 195 figures not in the passage.

## Predictions, scored

1. Right: sol 0.980 is inside 0.970–0.995, and luna's 0.973 and 0.966 are
   inside 0.955–0.990.
2. Half right:
   - Right: sol's passes differ by 0.0002 with 5 flips, inside "0.010 or
     less, at most 10 flips".
   - Wrong: luna's difference (0.007) is inside, but it flipped 29 cases,
     not 10 or fewer.
3. Right: the LoRA seeds score 0.908 and 0.912, inside 0.85–0.95, and differ
   by 0.004.
4. Right: sol beats the LoRA by 0.068–0.072, with intervals excluding zero.
5. Wrong: untuned Qwen scored 0.439, above 0.10–0.30.
6. Half right: the LoRA's top error is `missed`, as predicted, but untuned
   Qwen's is also `missed` (896), not `wrong_value` (675).
7. Right: 0.765 and 0.763 GPU-hours per seed are inside 0.6–0.8.
8. Mostly right:
   - The at-load figure ($0.061) is just above the predicted $0.02–0.06.
   - One at a time ($0.58) is inside $0.30–0.70.
   - Both are below sol; at load it is below luna and one at a time above
     it, as predicted.
9. Right: the LoRA's p50 (1.9 s) is below sol's (2.1–2.2 s) and luna's
   (2.4–2.5 s). Its p95 (7.1 s) sits between them.
10. Right: $4.70 total, inside $4–6.

## Spend

| item | amount | source |
|---|---|---|
| sol, 2 × 532 calls | $1.921 | reported tokens × list price |
| luna, 2 × 532 calls | $0.229 | same |
| full-run pod, 172.1 minutes at $0.89/hr | $2.553 | pod timestamps (created 18:39:28, stopped 21:31:35 UTC) |
| **full run** | **$4.70** of the $7 cap | |
| pilot, for reference | $1.09 | `FINETUNE_PILOT.md` |

RunPod's billing read at 22:42 UTC, after both pods were terminated:
**$3.065 GPU + $0.072 disk = $3.137** across the pilot and full-run pods.
The GPU figure matches the timestamps ($0.512 + $2.553). Billing was still
catching up at 21:32, when it showed $2.27.

**Wrapping up:**
- The full-run pod could not restart on a GPU: its host had none free. It
  was started on CPUs from the console for about a minute ($0.46/hr), and
  both adapters were downloaded. Their sha256 matched the pod's copies:
  seed 17 `62c24520…`, seed 29 `95ed78b6…`. They are in the gitignored
  `artifacts/finetune/qwen3-8b-lora-s17` and `-s29`.
- Both pods were then terminated. A fresh read of each returns 404, and
  the account lists no pods.

**Project total:** $2.73 API ($0.58 pilot + $2.15 full run) + $3.14
RunPod = **$5.87**.

## The reading

On this task a frontier model with four worked examples is the more
accurate choice: 0.980 against 0.91, a gap ten times the run-to-run noise.

The fine-tune's case is cost under load. At $0.06 per 1,000 requests with
the GPU kept busy, it is about 3.5× cheaper than gpt-6-luna and 25–34×
cheaper than gpt-6.1-sol, and its median latency is the lowest of the
three. One request at a time on a dedicated $0.89/hr GPU, it costs $0.58
per 1,000, more than luna.

So the fine-tune pays only at sustained volume, and even then it gives up
about 7 points of F1 and invents a few facts the API models never did
(13 figures its passages do not print; sol has none outside the label
artifacts below).

## Why the fine-tune loses

Read on 17 Oct 2026 from the stored answers. There were no model calls, and
the scorer, the labels and every reported score are unchanged. Each error
item `diagnose` counts for the LoRA (seed 17) was read beside sol's (pass 1)
answer to the same passage and given one cause.

- **Where the counts live:** `reports/finetune/full/lora-s17-error-causes.json`
  lists every item with its cause, the line it came from and sol's answer.
  Every count in this section comes from it.
- **How it is made:** `edgar-intel finetune causes` regenerates it, from
  `src/edgar_intel/finetune/causes.py`.
  - The causes are a reading, so each assignment must pass a rule on the
    passage text. For example, a parent-company cause needs "Parent Company
    Only" printed; an assignment that breaks its rule stops the run.
  - Omissions are split into prose and table by rule alone: prose when the
    line printing the figure has no table cell.
- **The unit:** an *item* is one count in the diagnosis. The 322 items are
  the LoRA's 139 missed, 101 unlabelled, 48 wrong_value, 18 wrong_year and
  16 not_in_passage. A wrong value always comes with a missed label for the
  same fact, so one wrong figure is two items.
- **"F1 if fixed":** the LoRA's micro F1 with only those items right: a
  missed label reported, and a figure no label has withdrawn. Its actual
  F1 is 0.908.

### The answer

**290 of the 322 items are the model's, and sol gets 270 of those right on
the same passages.** The other 32 are label artifacts. They cost sol 33 of
its own 70 items, so they are not why the LoRA loses. Set the 26 affected
(case, concept, year) keys aside for both arms and both scores rise by
0.0094, to 0.9177 against 0.9892. The gap is 0.0715 either way.

The model's errors fall into four groups, plus a few wrong cells:

| group | items | sol right | LoRA F1 if fixed |
|---|---|---|---|
| definitions the prompt states | 120 | 110 | 0.941 |
| which year a figure belongs to | 65 | 61 | 0.926 |
| figures left out | 74 | 70 | 0.931 |
| figures the passage does not print | 21 | 21 | 0.914 |
| wrong row or column | 10 | 8 | 0.911 |
| **all model errors** | **290** | **270** | 0.990 |
| label artifacts | 32 | 1 | 0.919 |

Fixing only the 270 items sol gets right would take the LoRA to 0.985,
above sol's 0.980.

**In plain English:**
- **Definitions (the largest group).** The LoRA doesn't apply the rules the
  prompt spells out: noncontrolling interests, Costco's membership fees,
  company totals rather than segments or the parent company.
- **Years.** It misplaces years where the table's shape, not a header next
  to the figure, says which year a figure is.
- **Omissions.** It leaves out figures that sit in unusual places: shares
  inside a row label, a row named "Earnings from operations", a figure in a
  sentence.
- **Unprinted figures.** It writes figures the passage doesn't print, which
  sol never does.
- **Scale.** Scale handling, the untuned model's main failure, is gone. One
  item in 322 looks like a scale slip: Adobe's printed "240" (millions)
  given as 2,400,000,000 (`scale_slips`).

### Every cause

| cause | items | missed | unlabelled | wrong value | wrong year | not in passage | sol right | companies |
|---|---|---|---|---|---|---|---|---|
| noncontrolling interests | 57 | 3 | 51 | 3 | 0 | 0 | 55 | CMCSA 27, PG 19, SLB 11 |
| net sales as total revenue | 26 | 6 | 14 | 6 | 0 | 0 | 26 | COST 26 |
| parent-company statements | 24 | 0 | 24 | 0 | 0 | 0 | 18 | UNH 24 |
| segment table | 13 | 3 | 6 | 3 | 0 | 1 | 11 | NVDA 8, ADBE 3, UNH 2 |
| year from the wrong header | 32 | 14 | 0 | 9 | 9 | 0 | 28 | UNH 8, NVDA 8, LMT 4, COST 4, PG 4, V 4 |
| multi-year row misread | 26 | 16 | 0 | 10 | 0 | 0 | 26 | LMT 20, NVDA 4, JNJ 2 |
| year guessed | 7 | 0 | 0 | 0 | 7 | 0 | 7 | LMT 5, ADBE 1, UNH 1 |
| shares outstanding | 24 | 22 | 0 | 2 | 0 | 0 | 24 | UNH 6, NVDA 6, LMT 6, ADBE 4, COST 2 |
| "Earnings from operations" skipped | 9 | 9 | 0 | 0 | 0 | 0 | 9 | UNH 9 |
| table figure omitted | 31 | 31 | 0 | 0 | 0 | 0 | 27 | 8 companies |
| prose figure omitted | 10 | 10 | 0 | 0 | 0 | 0 | 10 | LMT 7, COST 2, UNH 1 |
| digits added to a rounded figure | 8 | 4 | 0 | 4 | 0 | 0 | 8 | LMT 6, V 2 |
| figure not printed | 13 | 0 | 0 | 0 | 1 | 12 | 13 | PG 5, JNJ 3, ADBE 3, UNH 1, COST 1 |
| wrong row or column | 10 | 3 | 4 | 3 | 0 | 0 | 8 | PG 4, UNH 2, ADBE 2, NVDA 1, LMT 1 |
| *label:* beyond rounding | 16 | 8 | 0 | 8 | 0 | 0 | 0 | LMT 16 |
| *label:* restated figure | 6 | 0 | 2 | 0 | 1 | 3 | 0 | NVDA 3, JNJ 3 |
| *label:* year not printed | 10 | 10 | 0 | 0 | 0 | 0 | 1 | LMT 5, UNH 2, V 2, ADBE 1 |
| **total** | **322** | **139** | **101** | **48** | **18** | **16** | **271** | |

**Definitions the prompt states: 120 items, sol right on 110.**

- **Noncontrolling interests, 57 (sol right 55).** The prompt says net
  income and equity exclude noncontrolling interests. The LoRA reports:
  - Comcast's "Total equity | $ | 83,226 | $ | 81,627 | $ | 97,490" as
    stockholders' equity (9 items)
  - the "Net income (loss) | (15) | 9 | 25" rows under Comcast's
    noncontrolling-interest sections as net income (9)
  - the net income line that includes the interests (33), in:
    - comprehensive-income statements (Comcast, P&G)
    - cash-flow statements (SLB)
    - P&G's segment "Total Company" column
  - P&G's "Net earnings | 14,974" where the next row is "Net earnings
    attributable to Procter & Gamble | 14,879", and SLB's equity-statement
    total column (6)

  sol makes the same mistake twice, on P&G's segment net earnings.
- **Net sales as total revenue, 26 (sol right 26). All Costco.**
  - The prompt names this case: where net sales and membership fees are
    printed separately, total revenue is their sum.
  - The LoRA takes "Net sales | $ | 269,912" where the statement goes on to
    "Total revenue | 275,235" (12 items).
  - It also reports net sales as revenue in tables and sentences that print
    no total (14).
- **Parent-company statements, 24 (sol right 18).**
  - UnitedHealth's Schedule I is headed "Condensed Financial Information
    of Registrant (Parent Company Only)".
  - The LoRA reports its figures as the group's: cash of 776 against the
    group's 25,427, its total assets and liabilities, and "Total revenues |
    154 | 368 | 312".
  - sol does the same in one of the five passages (6 items).
- **Segment tables, 13 (sol right 11).**
  - Nvidia's "Operating income (loss) | $ | 5,083 | $ | 4,552 | $ | (5,411)
    | $ | 4,224" (segment, segment, all-other, total) is read as three years
    (5 items).
  - The total column of Nvidia's segment table is reported as operating
    income (3). That column sums the segments and is not the company's
    figure.
  - Adobe's segment gross profit is read as years (3, including the scale
    slip).
  - UnitedHealth's Optum subtotal is taken instead of the consolidated
    column (2).

**Which year a figure belongs to: 65 items, sol right on 61.**

- **Year from the wrong header, 32 (sol right 28).**
  - **Equity roll-forwards (16 items: Costco, Nvidia twice, P&G).** Activity
    after a year-end balance belongs to the next year. The LoRA gives the
    net income after "BALANCE AT AUGUST 28, 2022" to 2022, and the net
    income after "Balances, Jan 30, 2022" to fiscal 2022 rather than 2023.
  - **Statements stacked one block per year (12).** In UnitedHealth's
    segment blocks and Visa's EPS tables, the LoRA gives a row the year of
    the header below it, or reports a block printed under "2024" as 2025.
  - **Lockheed's second sales table (4)** sits under a printed year, "74
    2022" or "70 2023", and the LoRA gives it the year after.
  - **The 4 items sol doesn't get right** are wrong values on labels whose
    year the passage doesn't print. sol avoids the wrong figure, but can't
    report the label either.
- **Multi-year rows misread, 26 (sol right 26).**
  - **Lockheed cash-flow rows (20).**
    - In 17, the chunk starts below its year header, and the LoRA maps the
      columns to the wrong years, mostly reversed: in "Cash and cash
      equivalents at end of year | $ | 4,121 | $ | 2,483 | $ | 1,442" it
      gives 4,121 to 2023.
    - In 3, it takes the "at beginning of year" line for the year-end
      balance.
  - **J&J (2):** the same beginning-of-year line.
  - **Nvidia (4):** "Total shareholders' equity | 42,978 | 22,101" is given
    to the wrong years.
- **Year guessed, 7 (sol right 7).**
  - A sales or cash table that prints no year (Lockheed's sales by contract
    type, Adobe's cash note) is reported under the filing's own year, in all
    7.
  - sol leaves these out, which is what the prompt asks.
  - The labels on these figures are counted as the "year not printed"
    artifact below.

**Figures left out: 74 items, sol right on 70.**

- **Shares outstanding, 24 (sol right 24).** This is the LoRA's weakest
  concept (0.760 above).
  - It skips counts that sit inside a row label (12: UNH, COST, ADBE), as
    in "Common stock, $0.01 par value - 3,000 shares authorized; 906 and 915
    issued and outstanding | 9 | 9".
  - It skips the share column of equity-statement balances (6: NVDA), as
    in "Balances, Jan 29, 2023 | 2,466 | …".
  - Lockheed's note reads "Of the 242 million and 255 million shares of
    common stock issued and outstanding …, 240 million and 254 million
    shares were considered outstanding for consolidated balance sheet
    presentation purposes". The LoRA takes the first pair (4) or nothing
    (2).
- **"Earnings from operations" skipped, 9 (sol right 9).**
  - In three UnitedHealth results tables, the LoRA reads revenue, net
    earnings and EPS but skips "Earnings from operations | 32,358 | 28,435
    | 23,970".
  - In a fourth table it reads the same row. The same row is also among
    the segment-block omissions below.
- **Table figure omitted, 31 (sol right 27).** No single cause.
  - 6 are the oldest column of a row whose other years it read.
  - The rest include:
    - Lockheed's cash-flow summary row "Net earnings | 6,920 | 5,732 |
      6,315" (3)
    - Adobe's revenue total "Revenue | $ | 19,409 | $ | 17,606 | $ | 15,785"
      (3)
    - revenue and operating-income rows in UnitedHealth's segment blocks
      (4)
  - Costco's "TOTAL EQUITY | 23,622 | 25,058" sits in passages that also
    show a noncontrolling-interest column. sol leaves it out too (4).
- **Prose figure omitted, 10 (sol right 10).**
  - The EPS in Lockheed's "We reported net earnings of $6.9 billion ($27.55
    per share) in 2023" (4).
  - The R&D in "totaled $1.6 billion, $1.5 billion and $1.7 billion in
    2024, 2023 and 2022" (3).
  - Three more, at Costco and UnitedHealth.

**Figures the passage does not print: 21 items, sol right on all 21.**

- **Digits added to a rounded figure, 8 (sol right 8).**
  - For "$6.9 billion" the LoRA writes 6,866,000,000; for "$12.0 billion",
    12,047,000,000.
  - The label is the exact value, 6,920,000,000. sol's 6,900,000,000 is
    within the scorer's 0.5% of it; the LoRA's invented digits are not.
  - **The training data rewards it.** The training set has 153 labels
    printed only as a rounded prose figure, and each target is XBRL's exact
    value.
  - The same habit on labels that no reading can match is counted under
    the label artifacts, not here.
- **Figure not printed, 13 (sol right 13).**
  - **Prior-year figures the passage does not print,** beside
    current-year ones it does (5):
    - J&J's prior-year sales, next to sentences like "worldwide sales
      increased 6.0% to $94.2 billion" (3 passages)
    - Costco's 2025 net sales next to "Net sales increased 10% to
      $297,247" (1)
    - UnitedHealth's 2022 cash next to its 2023 cash (1)
  - **P&G (5):** operating income, net earnings and prior-year net sales, in
    a passage that prints them only as percentages of net sales and as
    "Net sales increased $245 million to $84.3 billion".
  - **Adobe (3):** gross profit figures that appear nowhere.
  - Outside the label artifacts, sol reports no figure its passage does not
    print.

**Wrong row or column: 10 items, sol right on 8.**

- P&G's non-GAAP "Core EPS" column instead of "As Reported" (4). sol
  misses the 2026 pair too.
- UnitedHealth's fair-value table: the Level 1 column instead of the total
  (2).
- For cash and cash equivalents (1 each):
  - Adobe's "Cash | $ | 787" line
  - Adobe's "Total cash, cash equivalents and short-term investments"
- The LoRA's facts on 2 of the 105 no-fact passages are here (1 each), and
  both are misread printed lines:
  - "Corporate debt securities | $ | 18,504 | … | 18,526" as Nvidia's cash
  - Lockheed's pension line, "Total FAS (expense) income | $ | (924)", as
    operating income

### Label artifacts: 32 of the LoRA's items, 33 of sol's

| artifact | LoRA items | sol items | sol right on the LoRA's |
|---|---|---|---|
| label beyond rounding | 16 | 16 | 0 |
| restated figure not a label | 6 | 8 | 0 |
| year not printed | 10 | 9 | 1 |

- **Label beyond rounding.**
  - Eight Lockheed labels are printed only as a rounded prose figure, "cash
    and cash equivalents of $1.4 billion", and the label is XBRL's
    1,442,000,000.
  - The label builder accepts the rounded form. The scorer allows 0.5%,
    and 1.4 billion is 2.9% away, so no reading of the passage can match.
  - Every arm loses all 8, as a missed label and a wrong value each. These
    are all 8 of sol's wrong values.
  - 8 of the 63 held-out labels printed only in rounded prose are like
    this, and 26 of the 153 in the training set.
- **Restated figure not a label.**
  - **What happens:** a 10-K's oldest column can be a later restatement:
    - Nvidia's FY2025 10-K prints FY2023 diluted EPS as 0.17 after the
      10:1 split ("Diluted | $ | 2.94 | $ | 1.19 | $ | 0.17").
    - J&J's FY2023 10-K prints 2021 sales as 78,740 after the Kenvue
      separation.
  - **Why it scores wrong:** the label builder takes a year's value from
    that year's 10-K and the next one, never the current filing's
    restatement, so a correct reading scores as a false positive.
  - **The not_in_passage counts:** all 3 of sol's `not_in_passage` are
    these, and so are 3 of the LoRA's 16. The figures are printed, but
    `diagnose` treats numbers under 1 as never printed.
- **Year not printed.**
  - The figure is printed with no year above it. Examples: the first
    Lockheed sales table in a chunk, and a Visa EPS block whose year line is
    in the previous chunk.
  - The prompt asks only for years printed with a figure, yet the key wants
    these.
  - sol reports one of the ten, inferring the year from the order of
    UnitedHealth's blocks.

### By company, and prose against table

| company | items | sol right | LoRA F1 if fixed |
|---|---|---|---|
| LMT | 74 | 51 | 0.9297 |
| UNH | 64 | 55 | 0.9265 |
| COST | 41 | 37 | 0.9199 |
| PG | 36 | 32 | 0.9182 |
| NVDA | 31 | 28 | 0.9172 |
| CMCSA | 27 | 27 | 0.9154 |
| ADBE | 18 | 17 | 0.9134 |
| SLB | 13 | 13 | 0.9118 |
| JNJ | 10 | 7 | 0.9111 |
| V | 8 | 4 | 0.9105 |

- **Lockheed** has the most items. 21 of its 74 are label artifacts (16
  beyond rounding, 5 with no year printed), and 20 are cash-flow rows
  mapped to the wrong years.
- **UnitedHealth's** 64 include:
  - its parent-company schedule (24)
  - its "Earnings from operations" row (9)
  - its stacked segment blocks: wrong headers (8), the segment subtotal
    (2), and the year guesses and missing years beside them
- **Comcast** is one cause: noncontrolling interests, all 27. **Costco** is
  mostly one: net sales, 26 of its 41.
- **Prose against table:**
  - 42 items are figures printed in prose: 26 model errors, all of which
    sol gets right, and the 16 rounding artifacts.
  - 266 are table figures; sol gets 231 right.
  - 14 are figures printed nowhere.
