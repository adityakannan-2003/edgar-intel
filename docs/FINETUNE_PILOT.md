# Fine-tune benchmark: pilot plan and predictions

Written 16 Oct 2026, before any paid call or GPU rental. Nothing below has been
run. The predictions are scored against the pilot's results file afterwards,
right or wrong, and the pilot runs only with Aditya's OK on the budget.

## What the pilot is for

To measure cost, latency and the things that break before spending the full
budget. Its F1 numbers come from 50 cases and are not the comparison, and the
pilot adapter (300 examples, one epoch) is there to measure training speed,
not quality.

## Cases and arms

- **Cases:** the first 50 of the held-out set's fixed shuffle
  (`finetune predict --n 50`): 41 with facts, 9 without, all 10 held-out
  companies. The held-out set is `data/finetune/held_out.jsonl`, sha256
  `146778436fd5a42f…`, built at `e05f0a7`.
- **Input:** every arm gets the same system instruction (179 tokens) and the
  same user message: company, filing year, the years to report, and the
  passage (661 tokens on average).
- **Few-shot arms:** these also get four worked examples from training
  companies (CVS, HD, MRK, and a PYPL table with no facts), which makes a
  2,910-token system prompt.

| arm | cases | where |
|---|---|---|
| gpt-6-luna few-shot | 50 | OpenAI API |
| gpt-6.1-sol few-shot | 50 | OpenAI API |
| gpt-6-astra few-shot | 25 (the first 25 of the same 50) | OpenAI API |
| Qwen3-8B few-shot, no LoRA | 50 | vLLM on the rented GPU |
| Qwen3-8B + pilot LoRA | 50 | vLLM on the rented GPU |

## Budget (cap $5)

Token counts were measured with the Qwen3 tokenizer; OpenAI's will differ
somewhat. Reasoning tokens are unknown until the first call.

| item | estimate | worst case assumed |
|---|---|---|
| luna, 50 calls | $0.01–0.02 | $0.05 |
| sol, 50 calls | $0.13–0.40 | $0.66 (no cache hits, 500 reasoning tokens a call) |
| astra, 25 calls | $0.36–1.00 | $1.63 (same) |
| RTX 4090 on RunPod, about 1.5 hours | $0.51–1.11 | $1.50 (2 hours on Secure Cloud) |
| **total** | **$1.01–2.53** | **$3.84** |

Prices: OpenAI's pricing page on 16 Oct 2026 (input / cached input / output per
1M tokens: luna $0.10 / $0.01 / $0.50, sol $2.00 / $0.10 / $10.00, astra
$10.00 / $1.00 / $50.00). For the GPU, RunPod's RTX 4090 at $0.34/hr Community
or $0.74/hr Secure.

## Predictions

1. At least one GPT-6 arm rejects a request parameter (`temperature`,
   `max_tokens`) on its first call. A 400 response costs nothing, and the
   abort after five failures stops the run.
2. Pilot micro F1: sol 0.80–0.92, luna 0.65–0.85, Qwen3-8B without LoRA
   0.55–0.80. Astra is within 0.03 of sol on the 25 cases both run.
3. Parse failures: 0 for the API arms, at most 2 of 50 for untuned Qwen3-8B.
4. The prompted arms' most common error is `wrong_year` (a column read as the
   wrong year) or `unlabelled` (a segment or non-GAAP figure), not
   `not_in_passage`.
5. Cost per 1,000 requests: luna $0.14–0.40, sol $2.50–8.00, astra
   $14–40.
6. p50 latency: luna under 2 s, sol 2–6 s, astra 4–15 s.
7. QLoRA training on an RTX 4090 runs at 1,500–3,000 tokens a second, so the
   full run's two epochs (3.4M tokens) take 0.3–0.6 GPU-hours per seed.

## Decision rules, fixed now

- If an arm's first five calls all fail, stop, fix the request and rerun
  that arm. A failed call is never scored as an empty answer.
- Astra stays in the full run only if its pilot micro F1 beats sol's with a
  paired-bootstrap 95% interval that excludes zero. Otherwise it is dropped:
  it costs about five times as much as sol.
- The full-run budget is the measured pilot cost per request × 539 cases ×
  2 passes (for run-to-run noise) per API arm, plus the measured training
  speed × 2 seeds, plus serving time. That number goes to Aditya for a
  second OK.
- Pilot numbers are never quoted as the comparison.

## What Aditya does for the GPU half

1. Create a RunPod account, add about $10 of credit, and add your SSH public
   key in RunPod's settings. Keys and payment stay with you.
2. Start one RTX 4090 pod from a PyTorch/CUDA template with a 60 GB disk,
   then paste the pod's SSH command (host and port, not a secret) into chat.
3. Stop the pod when the pilot finishes. Billing runs until you do.

I then install the `finetune` extra and vLLM on the pod, train the pilot
adapter, and serve the base model plus the adapter. The Mac reaches the server
through an SSH tunnel, so nothing is exposed publicly, and the benchmark runs
from the Mac. That way every arm's latency includes the same kind of internet
round trip.

## Pilot v1 results, and what changed before v2

Written 17 Oct 2026, after the v1 API half and before the v2 re-run.

**Spend:** $0.456 at list prices, from reported tokens
(`reports/finetune/pilot/v1/api-report.json`). The rejected first call
cost nothing.

| arm, v1 | cases | micro F1 | precision / recall | $ per 1k | p50 / p95 |
|---|---|---|---|---|---|
| gpt-6.1-sol | 50 | 0.943 | 0.916 / 0.970 | $2.35 | 2.1 s / 5.4 s |
| gpt-6-luna | 50 | 0.944 | 0.936 / 0.953 | $0.21 | 3.0 s / 12.2 s |
| gpt-6-astra | 25 | 0.958 | 0.919 / 1.000 | $13.10 | 2.0 s / 4.4 s |

**Predictions, scored:**

1. Right: luna rejected `temperature=0`.
2. Wrong for sol and luna, which both scored above their predicted ranges
   (sol 0.943 against 0.80–0.92, luna 0.944 against 0.65–0.85). Right for
   astra, which came within 0.03 of sol (+0.023).
3. Right: 0 parse failures.
4. Wrong: the top error was `unlabelled` for every arm, not `wrong_year`,
   and `wrong_year` was 0 for all three.
5. Right for all three: sol $2.35, luna $0.21 and astra $13.10 per 1,000
   are inside the predicted ranges.
6. Wrong for luna: its p50 is 3.0 s. It reasons by default, 169 hidden
   tokens on average and up to 1,086. Right for sol and astra.
7. Not yet measured (GPU).

**The astra rule:** astra beat sol by 0.023 on the 25 shared cases, with a
95% interval of [0.000, 0.063]. That interval does not exclude zero, so
astra is out of the full run.

**What the errors showed.** I read the 18 distinct `unlabelled` predictions
across the three arms.

- 13 were the instruction's fault:
  - 10 noncontrolling-interest totals ("Net income including noncontrolling
    interests", UNH's and CMCSA's "Total equity").
  - 3 Costco "Total net sales" read as total revenue.

  XBRL's net income and equity exclude noncontrolling interests, and total
  revenue includes membership fees, but the instruction never said so.
- The rest were XBRL's gaps. JNJ and Visa tag no `StockholdersEquity` at
  all, so JNJ's correctly read "Total shareholders' equity | 81,544" counted
  against every arm. Seven of the 40 companies lack that tag.

**Changes for v2:**

- The instruction states the XBRL definitions.
- Each record lists the (concept, year) pairs XBRL has facts for, and a
  prediction outside them is set aside as unverifiable.
- A company with no facts in its sampled cases has no F1, where v1 printed
  PG's as 0.0.

The v2 held-out set (sha256 `481a2923…`, built at `df3dc36`) has the same
539 ids, passages, labels and dropped labels as v1; only the instruction
differs (`reports/finetune/dataset-v1-v2-check.json`), so the label audit
carries over. v1 and v2 scores are different exams and are never compared.

**v2 runs luna and sol only**, on the same 50 cases, because astra is out
under the rule above. Re-running it on v2 would give it a second chance the
rule does not allow. **Prediction for v2:** `unlabelled` falls to 5 or fewer
per arm, and sol and luna stay within 0.03 of each other.

## Pilot v2 results, and the v3 re-grade

**v2** (instruction with the XBRL definitions; held-out sha256 `481a2923…`),
the same 50 cases, `reports/finetune/pilot/v2/api-report.json`:

| arm, v2 | micro F1 | precision / recall | missed | unlabelled | $ per 1k | p50 / p95 |
|---|---|---|---|---|---|---|
| gpt-6.1-sol | 0.979 | 1.000 / 0.959 | 7 | 0 | $2.29 | 2.4 s / 10.0 s |
| gpt-6-luna | 0.970 | 1.000 / 0.941 | 10 | 0 | $0.21 | 2.4 s / 8.1 s |

Both v2 predictions were right: `unlabelled` fell to 0 for both arms, and
luna is within 0.03 of sol (−0.009, 95% interval [−0.033, 0.000]).

All 7 of sol's misses were figures on rows that name a variant of the
concept: Costco's "Net income including noncontrolling interests" (equal to
net income only in years the interests were zero), JNJ's "Net earnings from
continuing operations", SLB's "Pretax segment operating income", and NVIDIA
share counts from equity-statement columns. Labels on rows naming a segment,
noncontrolling interests, or continuing or discontinued operations are now
dropped (`627b616`). That covers 1.9% of held-out labels, none of the 50
audited.

**v3** (held-out sha256 `e8e92b02…`, built at `627b616`): 532 held-out
examples (427 with facts, 1,734 labels) and 1,743 training examples. Seven
held-out passages whose only labels were on such rows left the set. Every
other input is unchanged and all 50 audited labels are still labels
(`reports/finetune/dataset-v2-v3-check.json`). This is the set the full run
uses.

**The v2 answers re-graded on v3 labels**, with no model calls, on the 47
pilot cases still in the set (`reports/finetune/pilot/v2/api-report-v3-labels.json`):

| arm | micro F1 | precision / recall | missed |
|---|---|---|---|
| gpt-6.1-sol | 0.994 | 1.000 / 0.987 | 2 |
| gpt-6-luna | 0.984 | 1.000 / 0.969 | 5 |

The misses left are NVIDIA share counts from an equity-statement column
(both arms) and luna skipping Costco's "Total assets" row three times. A
fine-tune has little room above the frontier prompt on accuracy. Its case,
if it has one, is cost and latency.

**Spend so far:** v1 $0.456, v2 $0.125, total **$0.581** of the $5 cap. The
GPU half has not started.

## Pilot GPU half (17 Oct 2026)

**Pod:** one RTX 4090 on RunPod Secure Cloud, `8ftc88t9ux400t`, US-NC-1,
host CUDA 13.2, $0.89/hr per the create response. Community Cloud had no
4090s available at the time, per the catalog read. The pod ran
17:37:22–18:11:54 UTC: **34.5 minutes, $0.51** from those timestamps.
RunPod's pod-billing read showed no records yet. About 14 of those minutes
were idle after the first setup failed (see below). The pod is stopped, and
its 10 GB volume (code, data, adapter, logs) keeps billing while stopped.

**Training** (`reports/finetune/pilot/gpu/manifest.json`):

| | |
|---|---|
| data | 300 seeded examples from v3 training (321,134 tokens), one epoch |
| setup | Qwen3-8B, QLoRA r=16, 4-bit base, effective batch 16 |
| trainer runtime | 215.5 s, **0.06 GPU-hours** |
| speed | **1,490 tokens/s** |
| final loss | 0.146 |
| versions | torch 2.14.1, transformers 5.20.1, peft 0.21.2, trl 1.16.0, bitsandbytes 0.50.2 |

**Serving:** vLLM 0.31.0 served Qwen3-8B (bf16, 15.36 GiB) with the adapter
beside it. The Mac reached it through an SSH tunnel.

**All four arms on the same 47 cases, v3 labels**
(`reports/finetune/pilot/pilot-report-v3-labels.json`):

| arm | micro F1 | precision / recall | $ per 1k, one at a time | $ per 1k at load | p50 / p95 |
|---|---|---|---|---|---|
| gpt-6.1-sol few-shot | 0.994 | 1.000 / 0.987 | $2.30 | n/a | 2.4 s / 9.8 s |
| gpt-6-luna few-shot | 0.984 | 1.000 / 0.969 | $0.20 | n/a | 2.3 s / 6.7 s |
| Qwen3-8B + pilot LoRA | 0.786 | 0.799 / 0.774 | $0.50 | **$0.043** | 1.7 s / 6.2 s |
| Qwen3-8B few-shot, no LoRA | 0.141 | 0.133 / 0.151 | $0.57 | not measured | 1.7 s / 6.1 s |

- **GPU cost.** It is the $0.89/hr rate times the measured time:
  - *One at a time* uses mean latency.
  - *At load* uses 5.70 requests/s with 32 in flight over 200 requests
    (`throughput-lora-c32-n200.json`; at 16 in flight over 50 it was
    3.99/s).
  - On a $0.34/hr Community 4090 the same arithmetic gives $0.19 and
    $0.017. That is a catalog price, not a measured run.
- **Paired bootstrap against sol:**
  - LoRA −0.208, 95% interval [−0.306, −0.123].
  - Untuned Qwen −0.853, [−0.944, −0.750].
- **Untuned Qwen3-8B reads the right cells and botches the scale.** "16,286"
  in millions becomes 1,628,600,000,000, one or two zeros too many. That is
  112 `wrong_value` errors.
- **The 300-example adapter has mostly learned the conversion** (10
  `wrong_value`). It misses more than it gets wrong: 36 `missed`, 11
  `unlabelled`, 8 `not_in_passage`.

**Two setup failures, both fixed in the pod script:**
- The first setup run died on Ubuntu 24.04's refusal to `pip install`
  outside a venv (`faa2a1f`). My status check matched its own command line
  and reported the dead script as running, so the pod sat idle for about
  14 minutes.
- vLLM's first start could not find `ninja` on PATH (`85e9839`).

**Predictions, scored:**
- 2, untuned Qwen3-8B 0.55–0.80: **wrong** (0.141).
- 3, at most 2 of 50 parse failures for untuned Qwen: **right** (0 parse
  failures; 3 unusable facts).
- 7, training at 1,500–3,000 tokens/s: **wrong**, just under (1,490).
- 6, latency for the self-hosted arms: not predicted.

**Pilot total: $0.58 API + $0.51 GPU = $1.09 of the $5 cap.**

## Full-run budget (needs a second OK)

All figures come from the pilot's measured rates. The set is v3's 532
held-out cases.

| item | basis | estimate |
|---|---|---|
| sol few-shot, 2 passes | 1,064 calls × $2.30 per 1k | $2.45 |
| luna few-shot, 2 passes | 1,064 calls × $0.20 per 1k | $0.21 |
| LoRA training, 2 seeds | 1,743 examples × 1,070 tokens × 2 epochs ÷ 1,490 tokens/s = 0.70 h per seed | 1.4 GPU-h |
| serving and predictions | 2 adapters + untuned base on 532 cases, one at a time, about 2 s each; throughput at load | about 1.0 GPU-h |
| setup and slack | | about 0.4 GPU-h |
| GPU total | 2.8 GPU-h × $0.89 Secure (or $0.34 Community if available) | $2.49 ($0.95) |
| **total** | | **about $5.15, cap $7** |
