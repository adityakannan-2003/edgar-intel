# Serving the fine-tune: load, cost, and when it beats the API

Written 3 Nov 2026, before any paid call. The predictions below are scored
against the run's results files afterwards, right or wrong.

Project #2 (`FINETUNE_RUN.md`) measured a Qwen3-8B + LoRA extractor at
0.908 micro F1 against gpt-6.1-sol's 0.980, and costed it from a single
throughput reading. This run serves that model as a service, load tests it
at 1, 4, 16 and 32 users, prices it per 1,000 requests, and finds the monthly
volume where renting a GPU beats paying the API.

## What is served

- **Model:** Qwen3-8B (bf16) with the seed-17 adapter as `extract-s17`. The
  seed-29 adapter is loaded beside it, so the server starts with exactly the
  17 Oct command (`scripts/finetune_pod.sh serve s17 s29`); only `extract-s17`
  is called.
- **Adapters:** `artifacts/finetune/qwen3-8b-lora-s17` and `-s29` in the main
  checkout (gitignored). `adapter_model.safetensors` sha256 `62c245208c42…`
  (s17) and `95ed78b63faf…` (s29), the hashes recorded on 17 Oct.
- **Requests:** the v3 held-out set, 532 cases, sha256 `e8e92b026c3c…`. Every
  request is the one `finetune predict` sends for the s17 arm: the
  instruction, the passage, temperature 0, at most 1,024 tokens, thinking
  off. `request_bodies` builds them, and a test checks they equal what
  `predict` posts.

## GPU and stack

**GPU: one RTX 4090 (24 GB), Secure Cloud, $0.89/hr.**

Stock read at about 18:45 UTC (`list-gpu-types`, Secure Cloud, pods): the 4090 is in
stock (Low) in EUR-IS-1 on CUDA 13.2 hosts. Community Cloud has nothing in
stock. The other cards in stock:

| card | memory | $/hr (Secure) | why not first |
|---|---|---|---|
| RTX 4090 | 24 GB | 0.89 | **chosen** |
| A40 | 48 GB | 0.59 | fallback: cheaper and more memory, but slower per request and untested with this model |
| A100 80GB PCIe | 80 GB | 1.79 | twice the rent |
| RTX PRO 6000 | 96 GB | 2.49 | |
| H100 NVL | 94 GB | 3.19 | |
| H100 SXM | 80 GB | 3.99 | |

The 4090 is the card and price project #2 served on. With the same vLLM
version and flags, the accuracy re-check compares the served model with
itself, not with another GPU's arithmetic, and the throughput can be checked
against 17 Oct's reading (4.07 requests/s with 32 in flight). The RTX 5090
and L40S showed stock in one read and none in the next.

**If the 4090 is gone at create time:** the A40 at $0.59/hr, same plan. Its
predictions are listed separately below.

**Stack:**
- Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (the official
  "Runpod Pytorch 2.8.0" template, from `list-public-templates`), CUDA floor
  13.0 because vLLM 0.31.0's torch is built for CUDA 13.0.
- vLLM **0.31.0**, pinned to the 17 Oct version, installed by
  `scripts/finetune_pod.sh setup serve` (the training environment is
  skipped).
- Flags, unchanged from 17 Oct: `--enable-lora --max-loras 2
  --max-lora-rank 16 --max-model-len 8192 --gpu-memory-utilization 0.90
  --seed 0`. Prefix caching and chunked prefill are on by default.
- Only port 22 is exposed. The Mac reaches vLLM through an SSH tunnel, so
  the server is never public. The account has no SSH key registered, so the
  Mac's public key goes in the pod's `PUBLIC_KEY` setting; nothing on the
  account changes.

**What 17 Oct's server log already says** (`reports/finetune/full/gpu/
pod-logs/serve-s17-s29.clean.log`): after the 15.4 GiB of weights, the
4090 had **1.95 GiB, 14,160 tokens, of KV cache**. A request here is about
1,070 tokens (965 in, 103 out), so the server can hold about 13 at once.
At 16 users a few wait; at 32, about 19 wait. That is the card's capacity
limit for this workload, and the load test should show it.

## How it is measured

All load comes from the Mac, through the tunnel, as on 17 Oct. Latency
includes the network to the pod; the round trip is measured separately
(median of 20 `/health` calls) and reported beside it.

1. **Accuracy first:** `edgar-intel finetune predict --arm qwen3-8b-lora-s17`
   over all 532 cases, one at a time, into
   `reports/serving/qwen3-8b-lora-s17.jsonl`. Then `finetune compare` with
   17 Oct's s17 file first, so the report gives the paired-bootstrap
   difference, its 95% interval and the cases flipped.
2. **Load test:** `edgar-intel bench sweep --arm qwen3-8b-lora-s17
   --endpoint /v1/chat/completions --concurrency 1,4,16,32
   --requests-per-level 200 --gpu-usd-per-hour 0.89`, into
   `reports/serving/sweep-rtx4090.json`. It reuses `run_bench` and `sweep`;
   each level replays the first 200 held-out requests (their 17 Oct latency,
   2.36 s mean, matches the full set's 2.34 s).
   - 32 is beyond the 1/4/16 asked for. It costs about a minute and shows
     whether 16 users already fill the card.
   - Prefix caching stays on, as it would in production. Each level sends
     200 requests and the cache holds about 13, so a level never finds its
     own requests still cached from the level before.
3. **Cost per 1,000 requests** at each level: the rent for the level's wall
   time divided by its successful requests (`gpu_usd_per_hour` in the
   sweep).

**"Matches 0.91 within noise"** means both:
- the served micro F1 is within 0.007 of 17 Oct's 0.9083 (0.007 is the
  largest same-configuration spread project #2 measured), and
- the paired-bootstrap 95% interval of the difference contains zero.

## Break-even method

A rented GPU costs the same whether it is busy or idle; the API costs per
request. With the GPU on all month (730 hours):

- **break-even volume** = monthly rent ÷ the API's cost per request.
  - It depends on the rent and the API price, not on the GPU's speed.
  - The speed sets how many GPUs a volume needs: one GPU carries up to its
    measured requests per second × 2.63 million seconds a month.
- **Above break-even**, each extra request costs the GPU's at-load cost
  (cents per 1,000) instead of the API's. That holds as long as the at-load
  cost is below the API's.

**The API prices compared**, all for this same extraction task:

| model | $ per 1,000 requests | source |
|---|---|---|
| gpt-6.1-sol few-shot (project #2's benchmark) | $1.54–2.07 | measured, `reports/finetune/full/report.json`, two passes |
| gpt-6-luna few-shot | $0.215 | measured, same file |
| gpt-4o-mini few-shot (serves /ask today) | $0.29–0.49 | **priced, not run**: the API arms' 2,928 prompt tokens and about 84 answer tokens at $0.15 / $0.075 cached / $0.60 per million. $0.29 if OpenAI caches the shared few-shot prompt as it did for luna (2,662 tokens), $0.49 if not |

gpt-4o-mini's accuracy on this task was never measured, so its row is a
price only. The fine-tune is 7 points of F1 below sol, so "beats the API"
below means on cost, at the fine-tune's accuracy.

## Spend and stop rules

- **Projected: about $1.05** (70 minutes at $0.89/hr): about 15 minutes of
  setup, 21 for the accuracy pass, 12 for the load test, and the rest for
  copying logs off and slack. No API calls.
- **Budget asked: $3**, against the $3–5 expected and the $15 hard cap.
- The pod is **terminated**, not stopped, as soon as the measurements and
  logs are off it. A fresh `get-pod` and `list-pods` confirm it is gone, and
  the billing read is recorded.
- If vLLM is not answering 45 minutes after the create, or the pod has run
  2 hours, it is terminated and the run is reported as it stands.

## Predictions

On the RTX 4090, 200 requests per level, measured from the Mac:

1. **Setup:** from the create call to vLLM answering `/v1/models` takes
   10–25 minutes.
2. **KV cache:** vLLM reports 13,500–14,900 tokens (17 Oct: 14,160).
3. **1 user:** 0.37–0.47 requests/s; p50 1.8–2.3 s; p95 6.5–8.5 s.
4. **4 users:** 1.2–1.8 requests/s; p50 2.0–2.9 s; p95 7–10 s.
5. **16 users:** 3.2–4.6 requests/s; p50 3.0–4.5 s; p95 9–13 s.
6. **32 users:** throughput within 15% of 16 users' (the KV cache is
   full), while mean latency roughly doubles (1.7–2.3×).
7. **Cost per 1,000 requests:** $0.054–0.077 at 16 users; $0.53–0.67 at 1
   user.
8. **Accuracy:** micro F1 0.901–0.915; the interval against 17 Oct contains
   zero; at most 15 cases flip; at least 90% of the 532 answers are
   byte-identical to 17 Oct's.
9. **Break-even** (arithmetic on the prices above, $649.70 a month for an
   always-on $0.89/hr GPU): sol 314k–421k requests a month, gpt-4o-mini
   1.33M–2.24M, luna 3.02M. Predicted: one 4090's capacity at 16 users
   (8.4M–12.1M a month) is above all three, so one GPU covers the volume
   where self-hosting starts to win.
10. **Spend:** $0.75–1.50 in RunPod billing.

**If the A40 is used instead:** 1 user p50 2.4–3.3 s; 16 users 3.0–5.0
requests/s with nobody waiting (48 GB holds far more than 16 requests);
$0.033–0.055 per 1,000 at 16 users; accuracy within 0.007 of 0.908 with at
most 40 cases flipped (different GPU arithmetic).

## Results (3 Nov 2026)

Every number below comes from a committed file in `reports/serving/`:

| file | what it holds |
|---|---|
| `qwen3-8b-lora-s17.jsonl` (+ `.meta.json`) | the 532 served answers, run at `88c63c0` |
| `accuracy-vs-17oct.json` | `finetune score` of 17 Oct's s17 file against today's, paired bootstrap |
| `sweep-rtx4090.json` | the load test |
| `breakeven.json` | `scripts/serving_breakeven.py` on the sweep and project #2's `report.json` (sha256 recorded inside) |
| `pod.json`, `pod-logs/` | the pod's reads, timeline and server log |

**The run:**
- **Pod:** `g9gwfi6zlxlobe`, one RTX 4090, Secure Cloud, $0.89/hr, created
  18:51:59 UTC.
  - The first create, pinned to EU-RO-1, was refused because the card had
    gone; a retry without a data center was placed.
  - vLLM 0.31.0 (torch 2.13.0+cu130, the same as 17 Oct) answered at
    19:18:11.
  - Terminated at 19:48:5x; `get-pod` returns 404 and `list-pods` is empty.
- **This card has ECC off: 24,564 MiB, against 23,028 on 17 Oct.**
  - vLLM got 3.29 GiB of KV cache, 23,920 tokens, against 1.95 GiB and
    14,160.
  - About 22 requests of this size fit at once, not 13. Prefix caching
    shares the instruction between requests, so up to 29 ran at once.
- **The network:** the Mac was 165 ms from the pod (median of 20 `/health`
  calls through the tunnel). Every latency below includes it.

### Accuracy: the served model is the 17 Oct model

| | micro F1 | precision / recall | facts on the 105 no-fact passages | p50 / p95, one at a time |
|---|---|---|---|---|
| 17 Oct, s17 | 0.9083 | 0.897 / 0.920 | 2 | 1.85 s / 7.09 s |
| 3 Nov, served s17 | **0.9100** | 0.899 / 0.921 | 2 | 1.72 s / 6.37 s |

- **The difference** is +0.0017, with a 95% interval of [−0.0007, +0.005].
  That is within 0.007 and contains zero, so the served model matches 0.91
  within noise.
- **519 of the 532 answers are byte-identical** to 17 Oct's. 13 differ in
  text, and 1 case changes whether its facts exactly match the key.
- **Error counts move by one to three:** missed 139 → 137, wrong value
  48 → 45, unlabelled 101 → 100.

### Load (`sweep-rtx4090.json`)

200 requests per level, replaying the first 200 held-out requests. No
request failed at any level.

| users | requests/s | p50 | p95 | p99 | $ per 1,000 |
|---|---|---|---|---|---|
| 1 | 0.47 | 1.75 s | 6.37 s | 7.84 s | $0.530 |
| 4 | 1.63 | 1.98 s | 6.89 s | 9.09 s | $0.151 |
| 16 | 4.22 | 2.74 s | 9.05 s | 12.72 s | **$0.0585** |
| 32 | 5.46 | 3.98 s | 12.28 s | 17.62 s | $0.0453 |

- **1 to 16 users:** throughput grows 9× while the median grows 1.6×. With
  one user the GPU is mostly idle.
  - The server log shows every request running at 16 users (14–16 running,
    none waiting), with the KV cache 50–53% full.
- **16 to 32 users:** throughput grows 29% and p95 36%.
  - The card is full at 32: 27–29 running, 1–3 waiting, KV cache 95–99%.
  - The sweep's knee rule (p95 up 1.5× while throughput grows under 1.2×)
    did not fire, so there is no hard knee between 1 and 32. Even so, about
    5.5 requests/s is this card's ceiling for this workload. Beyond 32
    users, extra requests would mostly wait.
- **Against 17 Oct:** 5.46 requests/s at 32 users, against 4.07. The
  difference is the larger KV cache.
- **The p95 is long at every level,** 6.4 s even for one user. A few
  passages produce 300–500-token answers, so p95 tracks answer length more
  than load.

### Cost and break-even (`breakeven.json`)

An always-on RTX 4090 at $0.89/hr costs **$649.70 a month**. One card
carries 11.1 million requests a month at the 16-user rate, or 14.3 million
at 32.

| API, same task | $ per 1,000 | self-hosting is cheaper above | per day | GPU at 16 users is cheaper per request by |
|---|---|---|---|---|
| gpt-6.1-sol few-shot | $1.54–2.07 | **314k–421k** a month | 10k–14k | 26–35× |
| gpt-4o-mini few-shot (priced, not run) | $0.29–0.49 | **1.33M–2.24M** a month | 44k–74k | 5–8× |
| gpt-6-luna few-shot | $0.215 | **3.02M** a month | 99k | 3.7× |

- **One card covers every break-even point.** At luna's 3.02M a month, the
  card would be busy 27% of the time at the 16-user rate.
- **Below 314k requests a month,** every API here is cheaper than keeping
  a GPU on. **From 314k to 3.02M,** the GPU beats sol, and beats
  gpt-4o-mini from 1.3–2.2M. **Above 3.02M,** it beats all three on cost.
- **Accuracy decides which comparison counts.** The fine-tune scores 0.91,
  against sol's 0.98 and luna's 0.97.
  - The comparison closest in quality is luna, which is still 6 points
    better and cheaper below about 3 million requests a month.
  - gpt-4o-mini's accuracy on this task has never been measured.
- **The rent is the lever:**
  - On a Community 4090 at its $0.34/hr list price (none was in stock
    today), the month is $248 and the luna break-even falls to 1.15M. That
    is arithmetic on a catalog price, not a measured run.
  - A server that runs only during busy hours pays less than the
    always-on figure. The at-load figure, $0.045–0.059 per 1,000, is the
    floor.

### Predictions, scored

1. **Wrong, just over.** Setup took 26.2 minutes (18:51:59 → 19:18:11),
   against 10–25.
   - The pod's own setup took 7 minutes.
   - Uploading the two 94 MB adapters from the Mac took 15, at about
     0.2–0.4 MB/s.
2. **Wrong.** The KV cache was 23,920 tokens, against 13,500–14,900. This
   card runs with ECC off, which leaves vLLM 1.3 GiB more for the cache.
   The same GPU model does not mean the same usable memory.
3. **Half right.**
   - 0.47 requests/s is inside 0.37–0.47, at the top.
   - The p50 (1.75 s) and p95 (6.37 s) are just below 1.8–2.3 s and
     6.5–8.5 s.
   - Each request was about 0.2 s faster than on 17 Oct (2.13 s against
     2.34 s mean, one at a time), for a reason not measured.
4. **Half right.** 1.63 requests/s is inside 1.2–1.8. The p50 (1.98 s) and
   p95 (6.89 s) are just below 2.0–2.9 s and 7–10 s.
5. **Mostly right.** 4.22 requests/s is inside 3.2–4.6 and the p95 (9.05 s)
   inside 9–13 s. The p50 (2.74 s) is below 3.0–4.5 s.
6. **Wrong.** 32 users added 29% throughput, where within 15% was
   predicted. Mean latency rose 1.47×, against 1.7–2.3×. Both follow from
   prediction 2's miss: with room for about 22 requests, not 13, 32 users
   still found room.
7. **Half right.** $0.0585 at 16 users is inside $0.054–0.077. $0.5296 at
   one user is a hair under $0.53–0.67.
8. **Right.**
   - F1 0.9100 is inside 0.901–0.915.
   - The interval [−0.0007, +0.005] contains zero.
   - 1 case flipped, against at most 15.
   - 97.6% of the answers are byte-identical, against at least 90%.
9. **Right.** The break-even volumes are the predicted arithmetic. One
   card's 16-user capacity, 11.1M a month, is inside 8.4M–12.1M and above
   all three.
10. **Right.** $0.84 from the pod's timestamps, inside $0.75–1.50.

Three right, one mostly right, three half right, three wrong. Every latency
miss (3–5) is a server faster than predicted, by a small margin. The two
clear misses (2, 6) share one cause: the card's memory, which a stock read
does not show.

### Spend

| item | amount | source |
|---|---|---|
| RTX 4090 pod, 56.9 minutes at $0.89/hr | $0.84 | pod timestamps (created 18:51:59, terminated 19:48:5x UTC) |
| API calls | $0 | none made |
| **this run** | **about $0.85** of the $3 cap, including a few cents of disk | |

RunPod's pod-billing read for `g9gwfi6zlxlobe` at about 19:50 UTC had no
records yet. On 17 Oct, billing lagged about an hour.

**After the run:**
- **Left billing:** nothing. The pod is terminated and no volume was
  created.
- **RunPod to date:** $3.14 (17 Oct) + about $0.85 = about $3.99.

## The reading

- **Serving reproduces the measured model.** 0.910 against 0.908, with 519
  of 532 answers byte-identical.
- **Capacity:** one $0.89/hr RTX 4090 serves 4.2 requests/s to 16
  concurrent users, at a 2.7 s median and a 9 s p95, and tops out near
  5.5/s.
- **Cost per 1,000 requests:** $0.53 one user at a time, $0.059 at 16
  users, $0.045 at 32.
- **Against the API:** an always-on GPU beats gpt-6.1-sol above about
  0.3–0.4 million requests a month, gpt-4o-mini above 1.3–2.2 million, and
  gpt-6-luna above 3 million. One card covers all of those volumes.
- **The catch is accuracy.** At 0.91 F1, against 0.97–0.98, the fine-tune
  pays only where a somewhat less accurate extractor is acceptable and
  volume runs to millions of requests a month. Below that, gpt-6-luna is
  both cheaper and more accurate.
