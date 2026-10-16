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
