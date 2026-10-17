"""Fine-tune vs. prompting benchmark.

The comparison that makes a fine-tuning project worth resume space is not
"I fine-tuned a model and got 89%". It is "I fine-tuned a model, compared it
against a prompting baseline on the same held-out set, and can tell you the
quality, cost and latency tradeoff in both directions."

That requires holding everything constant except the thing being compared:
same held-out examples, same instruction and input, same parser, same scorer.
The only differences between arms are the weights and, for prompted arms, the
worked examples in the system prompt.

Two phases, never one. `predict` sends each held-out case to one arm and
appends the raw output to a JSONL file as it arrives; `score` reads those
files. A paid run interrupted at case 400 keeps its 400 answers, a scoring
change re-scores without a single model call, and the raw outputs are the
record every reported number comes from.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .dataset import INSTRUCTION, TABLE_FIGURE, load_records, sha256_file
from .scoring import (
    CONCEPTS,
    diagnose,
    normalise_fact,
    parse_extraction,
    parse_ok,
    score_extraction,
)

# (input, cached input, output) USD per 1M tokens, standard tier, read from
# developers.openai.com/api/docs/pricing on 16 Oct 2026. Update with the date.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "gpt-6-astra": (10.00, 1.00, 50.00),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}


@dataclass(slots=True)
class Arm:
    name: str
    model: str
    few_shot: bool = True
    # Self-hosted arms (vLLM) need a base_url at predict time; API arms use the
    # configured provider. Keys are never stored here.
    self_hosted: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 1024
    notes: str = ""

    def config(self) -> dict[str, Any]:
        return asdict(self)


# The GPT-6 models reject `max_tokens` and any temperature but the default
# of 1. The pilot's first call, 17 Oct 2026, returned "'temperature' does not
# support 0.0 with this model" (reports/finetune/pilot/v1/
# luna-rejected-temperature.jsonl). At temperature 1 an answer can change
# between runs, which is what the full run's two passes per arm measure.
_OPENAI_REASONING = {"max_tokens": None, "max_completion_tokens": 2048, "temperature": None}
# Qwen3 thinks before answering unless told not to; extraction does not need it.
_QWEN_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

ARMS: dict[str, Arm] = {
    "luna-fewshot": Arm("gpt-6-luna few-shot", "gpt-6-luna", extra_body=_OPENAI_REASONING),
    "sol-fewshot": Arm("gpt-6.1-sol few-shot", "gpt-6.1-sol", extra_body=_OPENAI_REASONING),
    "astra-fewshot": Arm("gpt-6-astra few-shot", "gpt-6-astra", extra_body=_OPENAI_REASONING),
    "qwen3-8b-fewshot": Arm(
        "Qwen3-8B few-shot, no LoRA", "Qwen/Qwen3-8B", self_hosted=True,
        extra_body=_QWEN_NO_THINK,
        notes="isolates what the LoRA adds from what the base model already does",
    ),
    "qwen3-8b-lora": Arm(
        "Qwen3-8B + LoRA", "extract", few_shot=False, self_hosted=True,
        extra_body=_QWEN_NO_THINK,
        notes="vLLM serving the adapter under the name 'extract'",
    ),
}


# ------------------------------------------------------------------- shots
def pick_shots(
    train_path: str, n_pos: int = 3, max_chars: int = 1400, seed: int = 7
) -> list[dict[str, Any]]:
    """Worked examples for the prompted arms, from training companies only.

    Three positives from three different companies, chosen so that each adds
    concepts the others lack, plus one table that states no fact -- the case
    a prompted model most often gets wrong by inventing revenue from a
    segment table. Deterministic for a given training file and seed.
    """
    rng = random.Random(seed)
    rows = load_records(train_path)
    pos = [r for r in rows if 2 <= len(r["facts"]) <= 6 and len(r["passage"]) <= max_chars]
    neg = [
        r for r in rows
        if not r["facts"] and len(r["passage"]) <= max_chars
        and len(TABLE_FIGURE.findall(r["passage"])) >= 3
    ]
    rng.shuffle(pos)
    rng.shuffle(neg)

    chosen: list[dict[str, Any]] = []
    concepts: set[str] = set()
    for row in pos:
        new = {f["concept"] for f in row["facts"]} - concepts
        if row["ticker"] in {c["ticker"] for c in chosen} or not new:
            continue
        chosen.append(row)
        concepts |= new
        if len(chosen) == n_pos:
            break
    for row in neg:
        if row["ticker"] not in {c["ticker"] for c in chosen}:
            chosen.append(row)
            break
    return chosen


def system_prompt(shots: list[dict[str, Any]]) -> str:
    if not shots:
        return INSTRUCTION
    parts = [
        INSTRUCTION,
        "Worked examples follow. Each INPUT is what you will be given and each "
        "OUTPUT is the correct answer for it.",
    ]
    for shot in shots:
        parts.append(f"INPUT:\n{shot['messages'][1]['content']}\n\nOUTPUT:\n{shot['messages'][2]['content']}")
    return "\n\n".join(parts)


# ----------------------------------------------------------------- predict
def select(rows: list[dict[str, Any]], n: int | None, seed: int = 17) -> list[dict[str, Any]]:
    """The first-n cases of a fixed shuffle: every arm, and every pilot, gets
    the same cases, and a larger n is a superset of a smaller one."""
    if n is None or n >= len(rows):
        return rows
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[i] for i in sorted(order[:n])]


def input_sha256(row: dict[str, Any]) -> str:
    """A hash of what a model sees for one case -- instruction and user
    message -- and nothing else. Labels can then be corrected and stored
    answers re-scored without a model call, the way `eval regrade` re-marks
    stored answers."""
    return hashlib.sha256(json.dumps(row["messages"][:2]).encode()).hexdigest()


def inputs_sha256(rows: list[dict[str, Any]]) -> str:
    """One hash over every case's id and input, binding a predictions file to
    a held-out version for resuming."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row["id"], input_sha256(row)]).encode())
    return digest.hexdigest()


def _usage(raw: dict[str, Any]) -> dict[str, int]:
    usage = raw.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "cached_tokens": int(prompt_details.get("cached_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
    }


def predict(
    arm: Arm,
    held_out_path: str,
    out_path: str,
    *,
    train_path: str | None = None,
    n: int | None = None,
    base_url: str | None = None,
    llm: Any = None,
    max_consecutive_errors: int = 5,
    progress: Any = None,
) -> dict[str, Any]:
    """Run one arm over the held-out cases, appending each output as it arrives.

    Resumable: ids already in `out_path` are skipped. A file is bound to one
    configuration by its `.meta.json` sidecar, and predicting into it with a
    different arm, system prompt or held-out set is refused rather than
    silently mixed. Five consecutive call failures abort the run: 232
    consecutive HTTP 401s once scored as 0.0 in this repo.
    """
    rows = select(load_records(held_out_path), n)
    shots = pick_shots(train_path) if arm.few_shot else []
    if arm.few_shot and not shots:
        raise ValueError("a few-shot arm needs train_path to draw examples from")
    system = system_prompt(shots)

    meta = {
        "arm": arm.config(),
        "held_out_path": held_out_path,
        "held_out_sha256": sha256_file(held_out_path),
        "inputs_sha256": inputs_sha256(load_records(held_out_path)),
        "input_sha256_by_id": {r["id"]: input_sha256(r) for r in load_records(held_out_path)},
        "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "shot_ids": [s["id"] for s in shots],
        "n_requested": n,
    }
    meta_path = out_path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            existing = json.load(fh)
        keys = ("arm", "inputs_sha256", "system_sha256")
        if any(existing.get(k) != meta[k] for k in keys):
            raise ValueError(f"{out_path} holds another configuration; use a new file")
    else:
        from ..evals.runner import git_sha

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        meta["git_sha"] = git_sha()
        meta["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        meta["base_url"] = base_url
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    done: set[str] = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            done = {json.loads(line)["id"] for line in fh if line.strip()}

    if llm is None:
        from ..providers.openai_compat import OpenAICompatLLM

        if arm.self_hosted:
            if not base_url:
                raise ValueError(f"{arm.name} is self-hosted: pass the server's base_url")
            llm = OpenAICompatLLM(base_url=base_url, api_key="EMPTY", model=arm.model)
        else:
            llm = OpenAICompatLLM(model=arm.model)

    written = errors = consecutive = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for row in rows:
            if row["id"] in done:
                continue
            record: dict[str, Any] = {"id": row["id"]}
            started = time.perf_counter()
            try:
                completion = llm.complete(
                    row["messages"][1]["content"],
                    system=system,
                    model=arm.model,
                    temperature=0.0,
                    max_tokens=arm.max_tokens,
                    extra_body=arm.extra_body,
                )
            except Exception as exc:  # recorded per case, never scored as []
                record.update(error=str(exc)[:600], latency_ms=int((time.perf_counter() - started) * 1000))
                errors += 1
                consecutive += 1
            else:
                record.update(
                    text=completion.text,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    model=completion.model,
                    **_usage(completion.raw),
                )
                consecutive = 0
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            written += 1
            if progress:
                progress(f"{arm.name}: {written} written, {errors} errors")
            if consecutive >= max_consecutive_errors:
                raise RuntimeError(
                    f"{arm.name}: {consecutive} consecutive failures, aborted; last: {record['error']}"
                )
    return {"arm": arm.name, "cases": len(rows), "skipped": len(done), "written": written, "errors": errors}


def measure_throughput(
    arm: Arm,
    held_out_path: str,
    base_url: str,
    *,
    train_path: str | None = None,
    n: int = 50,
    concurrency: int = 16,
    llm: Any = None,
) -> dict[str, Any]:
    """Requests per second with `concurrency` requests in flight.

    This is the input to a self-hosted arm's cost at full load: GPU dollars
    per hour divided by requests per hour. Answers are discarded -- quality
    comes from `predict`, one request at a time, like every other arm.
    """
    from concurrent.futures import ThreadPoolExecutor

    rows = select(load_records(held_out_path), n)
    system = system_prompt(pick_shots(train_path) if arm.few_shot else [])
    if llm is None:
        from ..providers.openai_compat import OpenAICompatLLM

        llm = OpenAICompatLLM(base_url=base_url, api_key="EMPTY", model=arm.model)

    def one(row: dict[str, Any]) -> tuple[int, int]:
        started = time.perf_counter()
        completion = llm.complete(
            row["messages"][1]["content"], system=system, model=arm.model,
            temperature=0.0, max_tokens=arm.max_tokens, extra_body=arm.extra_body,
        )
        return int((time.perf_counter() - started) * 1000), completion.completion_tokens

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, rows))
    wall_s = time.perf_counter() - started
    latencies = [r[0] for r in results]
    return {
        "arm": arm.name,
        "base_url": base_url,
        "requests": len(rows),
        "concurrency": concurrency,
        "wall_s": round(wall_s, 2),
        "requests_per_s": round(len(rows) / wall_s, 3),
        "completion_tokens_per_s": round(sum(r[1] for r in results) / wall_s, 1),
        "latency_ms": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
    }


# ------------------------------------------------------------------- score
def _by_concept(facts: list[dict[str, Any]], concept: str) -> list[dict[str, Any]]:
    out = []
    for fact in facts:
        norm = normalise_fact(fact)
        if norm and norm[0] == concept:
            out.append(fact)
    return out


def score_case(record: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Score one prediction record against its held-out row."""
    case: dict[str, Any] = {
        "id": row["id"],
        "ticker": row["ticker"],
        "sector": row.get("sector", ""),
        "negative": not row["facts"],
        "latency_ms": record.get("latency_ms"),
        **{k: record.get(k, 0) for k in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens")},
    }
    if "error" in record:
        case["error"] = record["error"]
        return case
    predicted = parse_extraction(record["text"])
    verifiable = row.get("verifiable")
    case["parse_ok"] = parse_ok(record["text"])
    case.update(score_extraction(predicted, row["facts"], ignore=row["dropped"], verifiable=verifiable))
    case["diagnosis"] = diagnose(
        predicted, row["facts"], row["passage"], ignore=row["dropped"], verifiable=verifiable
    )
    case["by_concept"] = {}
    for concept in CONCEPTS:
        s = score_extraction(
            _by_concept(predicted, concept), _by_concept(row["facts"], concept),
            ignore=_by_concept(row["dropped"], concept), verifiable=verifiable,
        )
        if s["n_pred"] or s["n_expected"]:
            case["by_concept"][concept] = {k: s[k] for k in ("tp", "n_pred", "n_expected")}
    return case


def _micro(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled precision/recall/F1. None where there was nothing to score: a
    company whose sampled passages state no fact, answered [] each time, has
    no F1 -- the first pilot report printed it as 0.0."""
    tp = sum(c["tp"] for c in cases)
    n_pred = sum(c["n_pred"] for c in cases)
    n_exp = sum(c["n_expected"] for c in cases)
    p = tp / n_pred if n_pred else (None if not n_exp else 0.0)
    r = tp / n_exp if n_exp else (None if not n_pred else 0.0)
    if p is None or r is None:
        f1 = None
    else:
        f1 = round(2 * p * r / (p + r), 4) if p + r else 0.0
    return {"precision": None if p is None else round(p, 4),
            "recall": None if r is None else round(r, 4),
            "f1": f1, "tp": tp, "n_pred": n_pred, "n_expected": n_exp}


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def aggregate(
    cases: list[dict[str, Any]],
    model: str,
    gpu_usd_per_hour: float | None = None,
    throughput: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every reported number for one arm, from its scored cases."""
    scored = [c for c in cases if "error" not in c]
    negatives = [c for c in scored if c["negative"]]
    diag_totals: dict[str, int] = {}
    for c in scored:
        for k, v in c["diagnosis"].items():
            diag_totals[k] = diag_totals.get(k, 0) + v

    per_concept = {}
    for concept in CONCEPTS:
        parts = [c["by_concept"][concept] for c in scored if concept in c["by_concept"]]
        if parts:
            per_concept[concept] = _micro(parts)
    per_company = {
        t: {**_micro([c for c in scored if c["ticker"] == t]), "cases": sum(1 for c in scored if c["ticker"] == t)}
        for t in sorted({c["ticker"] for c in scored})
    }

    latencies = [c["latency_ms"] for c in scored if c.get("latency_ms") is not None]
    out: dict[str, Any] = {
        "cases": len(cases),
        "errors": len(cases) - len(scored),
        "micro": _micro(scored),
        "micro_positives_only": _micro([c for c in scored if not c["negative"]]),
        "macro_f1": round(statistics.fmean(c["f1"] for c in scored), 4) if scored else None,
        "exact_set_match": round(statistics.fmean(c["exact_set_match"] for c in scored), 4) if scored else None,
        "negatives": len(negatives),
        "negatives_with_any_fact": sum(1 for c in negatives if c["n_pred"] > 0),
        "parse_failures": sum(1 for c in scored if not c["parse_ok"]),
        "invalid_facts": sum(c["invalid"] for c in scored),
        "ignored_facts": sum(c["ignored"] for c in scored),
        "unverifiable_facts": sum(c.get("unverifiable", 0) for c in scored),
        "diagnosis": diag_totals,
        "per_concept": per_concept,
        "per_company": per_company,
        "latency_ms": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
        "tokens_mean": {
            k: round(statistics.fmean(c[k] for c in scored), 1) if scored else None
            for k in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens")
        },
    }

    if model in PRICES_USD_PER_MTOK:
        p_in, p_cached, p_out = PRICES_USD_PER_MTOK[model]
        total = sum(
            ((c["prompt_tokens"] - c["cached_tokens"]) * p_in + c["cached_tokens"] * p_cached
             + c["completion_tokens"] * p_out) / 1e6
            for c in cases
        )
        out["cost"] = {
            "basis": "list price on 16 Oct 2026, tokens as reported, errored calls included",
            "total_usd": round(total, 6),
            "per_1k_requests_usd": round(total / len(cases) * 1000, 4) if cases else None,
        }
    elif gpu_usd_per_hour is not None and latencies:
        mean_s = statistics.fmean(latencies) / 1000
        out["cost"] = {
            "basis": f"one request at a time on a ${gpu_usd_per_hour}/hr GPU, mean latency {mean_s:.2f}s",
            "per_1k_requests_usd": round(gpu_usd_per_hour * mean_s / 3.6, 4),
        }
        if throughput:
            rps = throughput["requests_per_s"]
            out["cost"]["at_load"] = {
                "basis": f"{throughput['concurrency']} requests in flight, {rps} requests/s measured",
                "per_1k_requests_usd": round(gpu_usd_per_hour / (rps * 3600) * 1000, 4),
            }
    else:
        out["cost"] = "NOT MEASURED"
    return out


def score(
    pred_path: str,
    held_out_path: str,
    gpu_usd_per_hour: float | None = None,
    throughput: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one predictions file against a held-out set with the same inputs.

    The labels may differ from the ones the file was predicted with -- that is
    a re-grade, and the report says so -- but the inputs may not: an answer to
    another question is not an answer to this one.
    """
    with open(pred_path + ".meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    rows = {r["id"]: r for r in load_records(held_out_path)}
    with open(pred_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    by_id = meta.get("input_sha256_by_id")
    if by_id is None:
        if meta["held_out_sha256"] != sha256_file(held_out_path):
            raise ValueError(f"{pred_path} was predicted on another held-out set")
    else:
        changed = [r["id"] for r in records if r["id"] in rows and by_id.get(r["id"]) != input_sha256(rows[r["id"]])]
        if changed:
            raise ValueError(
                f"{pred_path} was predicted on another held-out set: {len(changed)} inputs differ"
            )
    # A case that left the set (its only labels were dropped) is not scored;
    # a file sharing no case with this set was made on another one.
    left_set = [r["id"] for r in records if r["id"] not in rows]
    if records and len(left_set) == len(records):
        raise ValueError(f"{pred_path} was predicted on another held-out set: no case in common")
    cases = [score_case(rec, rows[rec["id"]]) for rec in records if rec["id"] in rows]
    return {
        "arm": meta["arm"]["name"],
        "pred_path": pred_path,
        "labels_sha256": sha256_file(held_out_path),
        "regraded": meta["held_out_sha256"] != sha256_file(held_out_path),
        "cases_no_longer_in_set": len(left_set),
        "meta": meta,
        "metrics": aggregate(cases, meta["arm"]["model"], gpu_usd_per_hour, throughput),
        "cases": cases,
    }


def paired_bootstrap(
    a: list[dict[str, Any]], b: list[dict[str, Any]], n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Micro-F1 difference b - a over the cases both scored, with a 95% interval.

    Resampling cases, not facts, because the facts in one passage are not
    independent: a model that misreads a table's column order misses all of
    them together.
    """
    a_by = {c["id"]: c for c in a if "error" not in c}
    b_by = {c["id"]: c for c in b if "error" not in c}
    ids = sorted(a_by.keys() & b_by.keys())
    if not ids:
        return {"cases": 0}
    def f1(cases: list[dict[str, Any]]) -> float:
        # A resample with no fact to find in either arm has no F1 in either;
        # counting both as 0 makes their difference 0, which is the truth.
        return _micro(cases)["f1"] or 0.0

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [rng.choice(ids) for _ in ids]
        diffs.append(f1([b_by[i] for i in sample]) - f1([a_by[i] for i in sample]))
    diffs.sort()
    return {
        "cases": len(ids),
        "f1_diff": round(f1([b_by[i] for i in ids]) - f1([a_by[i] for i in ids]), 4),
        "ci95": [round(diffs[int(0.025 * n_boot)], 4), round(diffs[int(0.975 * n_boot) - 1], 4)],
        "cases_flipped": sum(
            1 for i in ids if a_by[i]["exact_set_match"] != b_by[i]["exact_set_match"]
        ),
    }


def compare(
    pred_paths: list[str],
    held_out_path: str,
    out_path: str,
    gpu_usd_per_hour: dict[str, float] | None = None,
    throughput: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score every arm and compare each with the first, on the cases both have."""
    gpu, load = gpu_usd_per_hour or {}, throughput or {}
    results = [score(p, held_out_path, gpu.get(p), load.get(p)) for p in pred_paths]
    base = results[0]
    payload = {
        "held_out_path": held_out_path,
        "held_out_sha256": sha256_file(held_out_path),
        "arms": [
            {"arm": r["arm"], "pred_path": r["pred_path"], "regraded": r["regraded"],
             "cases_no_longer_in_set": r["cases_no_longer_in_set"], **r["metrics"]}
            for r in results
        ],
        "vs_first": [
            {"arm": r["arm"], "against": base["arm"], **paired_bootstrap(base["cases"], r["cases"])}
            for r in results[1:]
        ],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    with open(out_path.replace(".json", "") + ".cases.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            for case in r["cases"]:
                fh.write(json.dumps({"arm": r["arm"], **case}) + "\n")
    return payload


def table_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One flat row per arm, for the CLI table and the README."""
    rows = []
    for arm in payload["arms"]:
        cost = arm["cost"]
        rows.append({
            "arm": arm["arm"],
            "cases": arm["cases"],
            "errors": arm["errors"],
            "micro_f1": arm["micro"]["f1"],
            "precision": arm["micro"]["precision"],
            "recall": arm["micro"]["recall"],
            "negatives_with_a_fact": f"{arm['negatives_with_any_fact']}/{arm['negatives']}",
            "parse_failures": arm["parse_failures"],
            "p50_ms": arm["latency_ms"]["p50"],
            "p95_ms": arm["latency_ms"]["p95"],
            "usd_per_1k": cost["per_1k_requests_usd"] if isinstance(cost, dict) else cost,
            "usd_per_1k_at_load": cost.get("at_load", {}).get("per_1k_requests_usd")
            if isinstance(cost, dict) else None,
        })
    return rows
