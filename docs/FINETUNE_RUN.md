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

- **sol** fails only by omission: 40 `missed`, 17 `unlabelled`, 8
  `wrong_value`, 2 `wrong_year`. Its weakest concept is common shares
  outstanding (0.903); its weakest company is UNH (0.942).
- **The LoRA** misses more (139 `missed`) and reports figures that are not
  labels (101 `unlabelled`). It invents a few: 16 `not_in_passage`, and
  facts on 2 of the 105 no-fact passages, which no API arm did.
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

RunPod's billing read at 21:32 UTC showed $2.27 across both pods ($2.22 GPU,
$0.05 disk). That is less than the timestamps imply, so its records were
still catching up.

Both pods are stopped (`EXITED`, confirmed by a fresh read). Each keeps a
10 GB volume, which bills until it is terminated. The trained adapters
(87 MB each) are on those volumes; only their manifests and configs were
copied to the Mac.

## The reading

On this task a frontier model with four worked examples is the more
accurate choice: 0.980 against 0.91, a gap ten times the run-to-run noise.

The fine-tune's case is cost under load. At $0.06 per 1,000 requests with
the GPU kept busy, it is about 3.5× cheaper than gpt-6-luna and 25–34×
cheaper than gpt-6.1-sol, and its median latency is the lowest of the
three. One request at a time on a dedicated $0.89/hr GPU, it costs $0.58
per 1,000, more than luna.

So the fine-tune pays only at sustained volume, and even then it gives up
about 7 points of F1 and invents a few facts the API models never did.
