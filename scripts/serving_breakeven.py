"""Monthly volume where an always-on rented GPU beats the API (docs/SERVING.md).

    python scripts/serving_breakeven.py REPORT_JSON SWEEP_JSON OUT_JSON

REPORT_JSON is project #2's `reports/finetune/full/report.json` (the API arms'
measured cost and tokens); SWEEP_JSON is `edgar-intel bench sweep` output with
`gpu_usd_per_hour` set. Arithmetic only: no calls.
"""

from __future__ import annotations

import hashlib
import json
import sys

from edgar_intel.finetune.benchmark import PRICES_USD_PER_MTOK

HOURS_PER_MONTH = 730
SECONDS_PER_MONTH = HOURS_PER_MONTH * 3600


def main(report_path: str, sweep_path: str, out_path: str) -> dict:
    report = json.load(open(report_path))
    sweep = json.load(open(sweep_path))
    # Each API arm ran twice under the same name: keep both passes.
    sol = [a for a in report["arms"] if a["arm"].startswith("gpt-6.1-sol")]
    luna = [a for a in report["arms"] if a["arm"].startswith("gpt-6-luna")]
    assert len(sol) == len(luna) == 2, "expected two passes of each API arm"

    # gpt-4o-mini never ran this task: price the few-shot prompt luna was sent
    # (same prompt for every API arm) and its answer without reasoning tokens.
    p_in, p_cached, p_out = PRICES_USD_PER_MTOK["gpt-4o-mini"]
    tok = luna[0]["tokens_mean"]
    answer = tok["completion_tokens"] - tok["reasoning_tokens"]
    uncached = (tok["prompt_tokens"] * p_in + answer * p_out) / 1e6 * 1000
    cached = ((tok["prompt_tokens"] - tok["cached_tokens"]) * p_in
              + tok["cached_tokens"] * p_cached + answer * p_out) / 1e6 * 1000

    api = {
        "gpt-6.1-sol few-shot (measured)": sorted(a["cost"]["per_1k_requests_usd"] for a in sol),
        "gpt-6-luna few-shot (measured)": sorted(a["cost"]["per_1k_requests_usd"] for a in luna),
        "gpt-4o-mini few-shot (priced, not run)": [round(cached, 4), round(uncached, 4)],
    }
    rent = sweep["gpu_usd_per_hour"]
    monthly = rent * HOURS_PER_MONTH
    levels = {
        lv["concurrency"]: {
            "requests_per_s": lv["throughput_rps"],
            "usd_per_1k": lv["cost_per_1k_usd"],
            "capacity_per_month": round(lv["throughput_rps"] * SECONDS_PER_MONTH),
        }
        for lv in sweep["levels"]
    }
    best = max(levels.values(), key=lambda lv: lv["requests_per_s"])
    breakeven = {
        name: {
            "usd_per_1k": prices,
            # The dearer API price breaks even at the lower volume.
            "requests_per_month": [round(monthly / p * 1000) for p in reversed(prices)],
            "share_of_best_capacity": [
                round(monthly / p * 1000 / best["capacity_per_month"], 3) for p in reversed(prices)
            ],
            "gpu_cheaper_at_load": best["usd_per_1k"] < min(prices),
        }
        for name, prices in api.items()
    }
    out = {
        "inputs": {
            "report_json": report_path,
            "report_sha256": hashlib.sha256(open(report_path, "rb").read()).hexdigest(),
            "sweep_json": sweep_path,
            "gpu_usd_per_hour": rent,
            "hours_per_month": HOURS_PER_MONTH,
            "gpt_4o_mini_usd_per_mtok": [p_in, p_cached, p_out],
            "gpt_4o_mini_tokens": {"prompt": tok["prompt_tokens"], "cached": tok["cached_tokens"],
                                   "answer": round(answer, 1)},
        },
        "gpu_usd_per_month": round(monthly, 2),
        "levels": levels,
        "breakeven": breakeven,
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:4]), indent=2))
