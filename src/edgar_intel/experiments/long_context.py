"""RAG against long context, with the same model on the same golden set.

Each question is answered two ways: from the passages hybrid retrieval returns
(arm `rag`, the `eval run` pipeline), and from the whole 10-K the question is
about (arm `long_context`). The plan, the predictions and the results are in
docs/LONG_CONTEXT.md.

Both arms send the same request through the Responses API: `ANSWER_SYSTEM`,
then the context, then the question, with `ANSWER_TEMPLATE`'s text split
between the two user messages. Only the context differs. The model is a GPT-6
one because the configured gpt-4o-mini has a 128K window and the largest 10-Ks
here are about 140K tokens.

Caching: on GPT-6 models an implicit cache write lands at the end of the whole
request, which includes the question, so a shared document is reusable only
behind an explicit breakpoint. The long-context arm puts one after the filing.
The RAG arm sends explicit mode with no breakpoint, which writes nothing, so it
pays the plain input rate instead of a write surcharge on prompts that are
never reused. Questions are ordered by filing so each filing is written once
and read by the rest of its questions.

Results go to JSONL files, never to eval_runs: `/stats/eval` and `eval gate`
read the latest eval_runs row, and an experiment must not become what they
serve.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import time
from typing import Any

from .. import db
from ..config import get_settings
from ..evals.evidence import index_shape, labels_for
from ..evals.goldenset import _comparative_source_facts, _numeric_value_needles
from ..evals.judge import grade_numeric, is_abstention
from ..evals.runner import ANSWER_SYSTEM, ANSWER_TEMPLATE
from ..evals.schemas import ANSWER_SCHEMA, EvalCase
from ..providers.openai_compat import OpenAICompatLLM, parse_json_strict
from ..retrieval import metrics as m
from ..retrieval.search import DEFAULT_CONTEXT_MAX_CHARS, Hit, build_context_report, search

ARMS = ("rag", "long_context")

# (input, cache read, cache write at the default 30-minute TTL, output) USD per
# 1M tokens, the tier for requests up to 272K input tokens. Read from
# developers.openai.com/api/docs/pricing on 30 Oct 2026.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float, float, float]] = {
    "gpt-6-luna": (0.10, 0.01, 0.125, 0.50),
    "gpt-6.1-sol": (2.00, 0.10, 2.50, 10.00),
}
# Above this many input tokens a request is billed at roughly double. No 10-K
# here comes near it, so cost_usd refuses rather than misprices.
SHORT_CONTEXT_TIER = 272_000

MAX_CONSECUTIVE_ERRORS = 5
NARRATIVE_NOT_JUDGED = "narrative: not judged (judge kappa 0.42, docs/METRICS.md section 3)"


# ------------------------------------------------------------------ request
def split_prompt(context: str, question: str) -> tuple[str, str]:
    """`ANSWER_TEMPLATE` rendered, cut just before the question.

    The head holds the context and is the cacheable prefix; the tail is the
    question and the reply instructions. Joined, they are byte for byte the
    prompt `eval run` sends.
    """
    prompt = ANSWER_TEMPLATE.format(context=context, question=question)
    cut = prompt.rindex("\n\nQUESTION:\n") + 2
    return prompt[:cut], prompt[cut:]


def request_body(
    model: str,
    context: str,
    question: str,
    *,
    cache: bool,
    effort: str,
    max_output_tokens: int,
    cache_key: str | None = None,
) -> dict[str, Any]:
    head, tail = split_prompt(context, question)
    context_block: dict[str, Any] = {"type": "input_text", "text": head}
    if cache:
        context_block["prompt_cache_breakpoint"] = {"mode": "explicit"}
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": [context_block]},
            {"role": "user", "content": [{"type": "input_text", "text": tail}]},
        ],
        # Explicit mode in both arms: with no breakpoint it writes nothing.
        "prompt_cache_options": {"mode": "explicit"},
        "reasoning": {"effort": effort},
        "max_output_tokens": max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": ANSWER_SCHEMA["title"],
                "schema": ANSWER_SCHEMA,
                "strict": True,
            }
        },
        "store": False,
    }
    if cache and cache_key:
        body["prompt_cache_key"] = cache_key
    return body


def parse_response(data: dict[str, Any]) -> tuple[str, dict[str, int]]:
    """The output text and the token counts a Responses API reply reports."""
    text = "".join(
        part.get("text", "")
        for item in data.get("output") or []
        if item.get("type") == "message"
        for part in item.get("content") or []
        if part.get("type") == "output_text"
    )
    usage = data.get("usage") or {}
    inp = usage.get("input_tokens_details") or {}
    out = usage.get("output_tokens_details") or {}
    return text, {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cached_tokens": int(inp.get("cached_tokens") or 0),
        "cache_write_tokens": int(inp.get("cache_write_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "reasoning_tokens": int(out.get("reasoning_tokens") or 0),
    }


def cost_usd(model: str, usage: dict[str, int]) -> float:
    """List-price cost of one request. Cached reads and cache writes are
    reported apart from each other, and both are part of input_tokens."""
    if usage["input_tokens"] > SHORT_CONTEXT_TIER:
        raise ValueError(f"{usage['input_tokens']} input tokens: above the {SHORT_CONTEXT_TIER} price tier")
    p_in, p_read, p_write, p_out = PRICES_USD_PER_MTOK[model]
    uncached = usage["input_tokens"] - usage["cached_tokens"] - usage["cache_write_tokens"]
    return (
        uncached * p_in
        + usage["cached_tokens"] * p_read
        + usage["cache_write_tokens"] * p_write
        + usage["output_tokens"] * p_out
    ) / 1e6


def answer_from(text: str) -> str:
    """The `answer` field, or the raw text when the reply is not JSON (as
    `eval run` does: a formatting slip is not a wrong figure)."""
    try:
        return str(parse_json_strict(text).get("answer", "")).strip()
    except Exception:
        return text.strip()


def figure_in_context(case: EvalCase, context: str) -> bool | None:
    """Whether every figure the answer needs is printed in the context, in a
    filing-style rendering (METRICS section 5's method). It splits a failure
    into evidence never shown and evidence misread. None for narrative cases
    and for values with no usable rendering."""
    if case.kind != "numeric":
        return None
    if case.difficulty == "comparative":
        values: list[tuple[float, str | None]] = [(v, None) for _, v in _comparative_source_facts(case.notes)]
    else:
        values = [(case.expected_value, case.unit)] if case.expected_value is not None else []
    found = []
    for value, unit in values:
        # Three-digit forms ("199" for $199.21B) match exhibit years and
        # dividend tables, so only 4+ digit or decimal renderings count.
        needles = [n for n in _numeric_value_needles(value, unit) if "." in n or sum(ch.isdigit() for ch in n) >= 4]
        if not needles:
            return None
        found.append(any(n in context for n in needles))
    return all(found) if found else None


def grade(case: EvalCase, answer: str) -> dict[str, Any]:
    """Numeric cases by the XBRL grader `eval run` uses. Narrative cases are
    kept for reading: the judge's pass rate is not usable (kappa 0.42), and
    the configured judge's 128K window cannot hold the long-context arm's
    source anyway."""
    if case.kind != "numeric":
        return {"passed": None, "abstained": False, "rationale": NARRATIVE_NOT_JUDGED}
    passed, _, rationale = grade_numeric(case, answer)
    return {"passed": passed, "abstained": is_abstention(answer), "rationale": rationale}


# ------------------------------------------------------------------ context
def rag_context(case: EvalCase, strategy: str, use_rerank: bool) -> tuple[str, dict[str, Any]]:
    """What `answer_question` builds for `eval run`: the same search call and
    the same 20,000-character context. tests/test_long_context.py holds the
    two equal."""
    result = search(
        case.question,
        strategy=strategy,
        mode="hybrid",
        use_rerank=use_rerank,
        ticker=case.ticker,
        fiscal_year=case.fiscal_year,
    )
    retrieval: dict[str, Any] = {}
    if case.relevant_chunk_ids:
        retrieval = m.summarise(result.ids, set(case.relevant_chunk_ids))
    retrieval["retrieve_ms"] = result.stage_latency_ms.get("retrieve_ms", 0)
    report = build_context_report(result.hits, max_chars=DEFAULT_CONTEXT_MAX_CHARS)
    return report.text, retrieval


def filing_context(ticker: str, fiscal_year: int) -> tuple[str, str]:
    """(accession, context): every section of the one 10-K for this company
    and year, in filing order, tagged the way retrieved passages are."""
    rows = db.query(
        """
        SELECT f.id AS filing_id, f.accession, f.cik, s.id AS section_id, s.item, s.body
          FROM sections s
          JOIN filings f    ON f.id = s.filing_id
          JOIN companies co ON co.cik = f.cik
         WHERE co.ticker = %s AND f.fiscal_year = %s AND f.form = '10-K'
         ORDER BY s.ordinal
        """,
        (ticker.upper(), fiscal_year),
    )
    accessions = sorted({r["accession"] for r in rows})
    if len(accessions) != 1:
        raise ValueError(f"{ticker} FY{fiscal_year}: expected one 10-K, found {accessions}")
    hits = [
        Hit(
            chunk_id=f"section:{r['section_id']}", body=r["body"], filing_id=r["filing_id"],
            score=0.0, source="filing", cik=r["cik"], ticker=ticker.upper(),
            fiscal_year=fiscal_year, item=r["item"],
        )
        for r in rows
    ]
    return accessions[0], build_context_report(hits, max_chars=10**9).text


# ---------------------------------------------------------------------- run
def order_by_filing(cases: list[EvalCase]) -> list[EvalCase]:
    """Grouped by filing, set order kept within each, so one filing's
    questions run back to back while its cache entry is warm."""
    return sorted(cases, key=lambda c: (c.ticker or "", c.fiscal_year or 0))


def source_sha() -> str:
    """HEAD of the checkout this module was imported from, '-dirty' when it has
    uncommitted changes. Runs start in the main checkout so pydantic finds
    .env, where `git rev-parse` would name the main checkout's commit."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", here, "status", "--porcelain", "--untracked-files=no"], text=True)
    except Exception:
        return ""
    return sha + ("-dirty" if dirty.strip() else "")


def load_records(path: str) -> list[dict[str, Any]]:
    """One record per case; a resumed case's last record wins."""
    if not os.path.exists(path):
        return []
    by_id: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                by_id[record["case_id"]] = record
    return list(by_id.values())


def run_arm(
    cases: list[EvalCase],
    arm: str,
    model: str,
    out_path: str,
    *,
    effort: str = "medium",
    max_output_tokens: int = 25_000,
    golden_path: str = "",
    filings: list[tuple[str, int]] | None = None,
    llm: Any = None,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """Answer every case in one arm, appending a record per case as it lands.

    Resumable: cases already answered without an error are skipped. The file
    is bound to its configuration by a `.meta.json` sidecar, and a resume
    under a different one is refused rather than mixed into it.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, not {arm!r}")
    if model not in PRICES_USD_PER_MTOK:
        raise ValueError(f"no price for {model!r}; add it to PRICES_USD_PER_MTOK with the date")
    s = get_settings()
    if filings:
        wanted = {(t.upper(), y) for t, y in filings}
        cases = [c for c in cases if ((c.ticker or "").upper(), c.fiscal_year) in wanted]

    config: dict[str, Any] = {
        "arm": arm,
        "model": model,
        "api": "responses",
        "reasoning_effort": effort,
        "max_output_tokens": max_output_tokens,
        "golden_path": golden_path,
        "golden_sha256": _sha256_file(golden_path) if golden_path else "",
        "filings": sorted(f"{t.upper()}:{y}" for t, y in filings) if filings else "all",
        "n_cases": len(cases),
        "order": "by filing (ticker, fiscal_year), set order within",
    }
    if arm == "rag":
        index = index_shape(s.default_strategy)
        if index["chunks"] == 0 or index["embedded"] < index["chunks"]:
            raise RuntimeError(f"index '{s.default_strategy}' is empty or partly embedded: {index}")
        cases, evidence = labels_for(cases, s.default_strategy)
        config.update(
            strategy=s.default_strategy, mode="hybrid", use_rerank=s.use_rerank,
            k=s.retrieve_k, top_n=s.rerank_top_n, rrf_k=s.rrf_k, embed_model=s.embed_model,
            context_max_chars=DEFAULT_CONTEXT_MAX_CHARS, index=index, evidence_labels=evidence,
            cache_breakpoint=None,
        )
    else:
        config.update(context="every section of the case's 10-K, in order", cache_breakpoint="after the filing")
    _bind_meta(out_path, config)

    done = {r["case_id"] for r in load_records(out_path) if not r.get("error")}
    llm = llm or OpenAICompatLLM(model=model, timeout=300)
    current: tuple[tuple[str, int] | None, str, str] = (None, "", "")  # filing, accession, context
    consecutive_errors = 0
    records: list[dict[str, Any]] = []
    ordered = order_by_filing(cases)
    for i, case in enumerate(ordered, start=1):
        key = ((case.ticker or "").upper(), case.fiscal_year or 0)
        first_of_filing = i == 1 or key != ((ordered[i - 2].ticker or "").upper(), ordered[i - 2].fiscal_year or 0)
        if case.case_id in done:
            continue
        record: dict[str, Any] = {
            "case_id": case.case_id, "kind": case.kind, "difficulty": case.difficulty,
            "ticker": key[0], "fiscal_year": key[1], "arm": arm, "model": model,
            "first_of_filing": first_of_filing, "expected": case.expected,
        }
        started = time.perf_counter()
        try:
            retrieval: dict[str, Any] = {}
            if arm == "rag":
                context, retrieval = rag_context(case, s.default_strategy, s.use_rerank)
                accession = None
            else:
                if current[0] != key:
                    current = (key, *filing_context(*key))
                _, accession, context = current
            context_ms = int((time.perf_counter() - started) * 1000)
            body = request_body(
                model, context, case.question, cache=arm == "long_context",
                effort=effort, max_output_tokens=max_output_tokens, cache_key=accession,
            )
            called = time.perf_counter()
            data = llm._post("/responses", body)
            model_ms = int((time.perf_counter() - called) * 1000)
            text, usage = parse_response(data)
            answer = answer_from(text)
            record.update(
                accession=accession,
                context_chars=len(context),
                context_sha256=hashlib.sha256(context.encode()).hexdigest()[:16],
                # The retrieved context is kept for reading failures; the
                # long-context one is the filing, reloadable by accession.
                context_text=context if arm == "rag" else None,
                figure_in_context=figure_in_context(case, context),
                retrieval=retrieval,
                status=data.get("status"),
                incomplete_reason=(data.get("incomplete_details") or {}).get("reason"),
                raw_text=text,
                answer=answer,
                usage=usage,
                cost_usd=round(cost_usd(model, usage), 8),
                context_ms=context_ms,
                model_ms=model_ms,
                latency_ms=context_ms + model_ms,
                **grade(case, answer),
            )
            consecutive_errors = 0
        except Exception as exc:
            record["error"] = str(exc)[:500]
            consecutive_errors += 1
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        records.append(record)
        if progress:
            progress(i, len(ordered), record)
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            raise RuntimeError(f"{consecutive_errors} consecutive failed calls, last: {record['error']}")
    return records


def _bind_meta(out_path: str, config: dict[str, Any]) -> None:
    meta_path = out_path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta["config"] != json.loads(json.dumps(config)):
            raise ValueError(f"{out_path} was started under a different configuration; use a new --out")
        meta["resumes"].append({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_sha": source_sha()})
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        meta = {"config": config, "git_sha": source_sha(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "resumes": []}
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def _sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ------------------------------------------------------------------- report
def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% interval for a proportion; fair at the edges, where the normal
    approximation is not."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(centre - half, 4), round(centre + half, 4)]


def _pct(values: list[int], q: float) -> int | None:
    # The repo's convention (runner.summarise_run): the value at index int(q * n).
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else None


def _graded(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r["kind"] == "numeric" and not r.get("error")]


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if not r.get("error")]
    numeric = _graded(records)
    n_pass = sum(1 for r in numeric if r["passed"])
    out: dict[str, Any] = {
        "cases": len(records),
        "errors": len(records) - len(ok),
        "incomplete": sum(1 for r in ok if r.get("status") != "completed"),
        "numeric": len(numeric),
        "numeric_passed": n_pass,
        "numeric_accuracy": round(n_pass / len(numeric), 4) if numeric else None,
        "numeric_accuracy_ci95": wilson(n_pass, len(numeric)),
        "by_difficulty": {},
        # As eval run splits them: abstained, or wrong without abstaining.
        "abstention_rate": round(sum(r["abstained"] for r in numeric) / len(numeric), 4) if numeric else None,
        "wrong_answer_rate": (
            round(sum(1 for r in numeric if not r["passed"] and not r["abstained"]) / len(numeric), 4)
            if numeric else None
        ),
        "narrative_answered": sum(1 for r in ok if r["kind"] == "narrative"),
        # Failures split by whether the needed figures were printed in what the model read.
        "numeric_failures_by_evidence": {
            label: sum(1 for r in numeric if not r["passed"] and r.get("figure_in_context") is value)
            for label, value in (("figure_in_context", True), ("figure_absent", False), ("unknown", None))
        },
        "latency_ms": {
            "p50": _pct([r["latency_ms"] for r in ok], 0.5),
            "p95": _pct([r["latency_ms"] for r in ok], 0.95),
            "model_p50": _pct([r["model_ms"] for r in ok], 0.5),
            "model_p95": _pct([r["model_ms"] for r in ok], 0.95),
        },
    }
    for difficulty in sorted({r["difficulty"] for r in numeric}):
        group = [r for r in numeric if r["difficulty"] == difficulty]
        k = sum(1 for r in group if r["passed"])
        out["by_difficulty"][difficulty] = {"passed": k, "of": len(group), "ci95": wilson(k, len(group))}

    if ok:
        totals = {k: sum(r["usage"][k] for r in ok) for k in ok[0]["usage"]}
        cost = sum(r["cost_usd"] for r in ok)
        out["tokens_mean"] = {k: round(v / len(ok), 1) for k, v in totals.items()}
        out["cost"] = {
            "basis": "list price read 30 Oct 2026 x tokens as reported; errored calls are not counted",
            "total_usd": round(cost, 4),
            "per_1k_questions_usd": round(cost / len(ok) * 1000, 4),
        }
        hits = [r for r in ok if r["usage"]["cached_tokens"] > 0]
        out["cache"] = {
            "requests_with_cache_read": len(hits),
            "request_hit_rate": round(len(hits) / len(ok), 4),
            "token_hit_rate": round(totals["cached_tokens"] / max(1, totals["input_tokens"]), 4),
            "requests_with_cache_write": sum(1 for r in ok if r["usage"]["cache_write_tokens"] > 0),
            "first_of_filing": sum(1 for r in ok if r["first_of_filing"]),
            # Every question after the first on its filing should read the cache
            # (RAG never caches, so it has none to miss).
            "misses_after_first": sorted(
                r["case_id"] for r in ok
                if r["arm"] == "long_context" and not r["first_of_filing"] and r["usage"]["cached_tokens"] == 0
            ),
        }
    return out


def paired(a: list[dict[str, Any]], b: list[dict[str, Any]], n_boot: int = 4000, seed: int = 0) -> dict[str, Any]:
    """Numeric accuracy b - a on the cases both graded, with a paired
    bootstrap 95% interval, and the cases that flipped each way."""
    pa = {r["case_id"]: bool(r["passed"]) for r in _graded(a)}
    pb = {r["case_id"]: bool(r["passed"]) for r in _graded(b)}
    ids = sorted(pa.keys() & pb.keys())
    if not ids:
        return {"cases": 0}
    delta = {i: int(pb[i]) - int(pa[i]) for i in ids}
    rng = random.Random(seed)
    diffs = sorted(sum(delta[rng.choice(ids)] for _ in ids) / len(ids) for _ in range(n_boot))
    return {
        "cases": len(ids),
        "accuracy_diff": round(sum(delta.values()) / len(ids), 4),
        "ci95": [round(diffs[int(0.025 * n_boot)], 4), round(diffs[int(0.975 * n_boot) - 1], 4)],
        "passed_only_in_b": sorted(i for i in ids if delta[i] > 0),
        "passed_only_in_a": sorted(i for i in ids if delta[i] < 0),
    }


def recheck_figures(records: list[dict[str, Any]], golden_path: str) -> None:
    """Recompute `figure_in_context` with the current rule, in place: from the
    stored context for RAG, from the filing (database) for long context."""
    from ..evals.goldenset import load

    cases = {c.case_id: c for c in load(golden_path)}
    filings: dict[tuple[str, int], str] = {}
    for r in records:
        if r.get("error") or r["kind"] != "numeric":
            continue
        context = r.get("context_text")
        if context is None:
            key = (r["ticker"], r["fiscal_year"])
            if key not in filings:
                filings[key] = filing_context(*key)[1]
            context = filings[key]
        r["figure_in_context"] = figure_in_context(cases[r["case_id"]], context)


def report(paths: list[str], out_path: str, recheck: bool = False) -> dict[str, Any]:
    """Summaries of every results file, and every pair compared."""
    runs = {p: load_records(p) for p in paths}
    metas = {}
    for p in paths:
        with open(p + ".meta.json", encoding="utf-8") as fh:
            metas[p] = json.load(fh)
        if recheck:
            recheck_figures(runs[p], metas[p]["config"]["golden_path"])
    payload = {
        "figures_rechecked_at": source_sha() if recheck else None,
        "runs": [
            {"path": p, "config": metas[p]["config"], "git_sha": metas[p]["git_sha"], **summarise(runs[p])}
            for p in paths
        ],
        "pairs": [
            {"a": a, "b": b, **paired(runs[a], runs[b])} for a, b in itertools.combinations(paths, 2)
        ],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload
