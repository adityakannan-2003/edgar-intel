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
