"""Full dissection of one case: prompt in, answer out, verdict, and why.

The context probe answers "did the evidence reach the model". When the answer
is yes and the case still fails, every remaining hypothesis lives downstream of
retrieval, and they are only separable if you can read the literal bytes at each
boundary. So this module persists them:

  * every selected hit in final-prompt order, with body previews and Item
  * which gold chunk ids were included and which the cap dropped
  * the exact context string handed to ANSWER_TEMPLATE -- not a reconstruction
  * the exact rendered prompt and system message
  * the exact generated answer, the reference, and the judge's exact rationale
  * a literal search of the prompt text for each fact the reference requires

That last one matters more than it looks. It is tempting to read "4/5 relevant
chunks included" and conclude the facts are present. The gold label says those
chunks are relevant; it does not say they contain any particular sentence. The
only way to know whether the prompt mentions the DOJ lawsuit is to search the
prompt for the DOJ lawsuit.

Two replays close the loop, and they isolate opposite halves of the pipeline:

  `replay_generation` re-runs generation from the saved prompt with retrieval
  bypassed entirely. If it fails with the facts present, the problem is the
  answer prompt or the model.

  `replay_judge` re-runs the judge over the saved (question, reference, answer)
  triple. If a good answer is failed repeatedly, the problem is the judge, and
  no amount of retrieval work will move the number.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import get_settings
from ..providers import get_llm
from ..providers.openai_compat import parse_json_strict
from ..retrieval.search import DEFAULT_CONTEXT_MAX_CHARS, infer_expected_items
from .judge import JUDGE_SYSTEM, JUDGE_TEMPLATE, grade, judge_narrative
from .runner import ANSWER_SYSTEM, ANSWER_TEMPLATE, answer_question, git_sha
from .schemas import ANSWER_SCHEMA, EvalCase

BODY_PREVIEW_CHARS = 900


# ---------------------------------------------------------------- fact probes
def derive_fact_probes(reference: str) -> dict[str, str]:
    """Required facts, taken from the reference answer rather than hand-written.

    A hand-written probe list is a second place to encode the expected answer,
    and the two drift. These come out of the reference itself: named entities
    (runs of capitalised words), currency amounts, and statutory references.

    The result is a {label: regex} map. It is a starting point -- `--facts`
    overrides it when a case needs something this cannot see.
    """
    probes: dict[str, str] = {}

    # Currency amounts, including the euro sign the reference actually uses.
    for amount in re.findall(r"[€$£]\s?\d[\d,.]*\s*(?:million|billion|thousand)?", reference):
        cleaned = amount.strip()
        digits = re.sub(r"[^\d]", "", cleaned.split()[0])
        if not digits:
            continue
        scale = ""
        for word in ("million", "billion", "thousand"):
            if word in cleaned.lower():
                scale = rf"\s*(?:{word}|m\b|bn\b)?"
                break
        probes[cleaned] = rf"[€$£]?\s?{digits[:3]}[\d,.]*{scale}"

    # Statutory / regulatory references: "Article 5(4)", "Digital Markets Act".
    for article in re.findall(r"Article\s+\d+\(\d+\)", reference):
        probes[article] = re.escape(article).replace(r"\ ", r"\s+")

    # Named entities: two or more consecutive capitalised words, minus the
    # sentence-initial ones that are just grammar.
    stop = {"It", "The", "Apple", "This", "These", "Company"}
    # Leading qualifiers are how the same entity gets two spellings: the
    # reference says "EU Digital Markets Act", the answer says "Digital Markets
    # Act (DMA)". A probe that demands the qualifier reports a false NO, and a
    # false NO in a coverage table sends the next day's work in the wrong
    # direction.
    qualifiers = {"EU", "US", "U.S.", "UK", "European", "Federal", "State"}

    for phrase in re.findall(r"\b([A-Z][\w&.-]+(?:\s+[A-Z][\w&.-]+)+)\b", reference):
        words = phrase.split()
        if words[0] in stop:
            words = words[1:]
        while words and words[0] in qualifiers:
            words = words[1:]
        if len(words) < 2:
            continue
        core = " ".join(words)
        alternatives = [re.escape(core).replace(r"\ ", r"\s+")]
        # The acronym a reader would form from the same phrase.
        acronym = "".join(w[0] for w in words if w[0].isupper())
        if len(acronym) >= 3:
            alternatives.append(rf"\b{acronym}\b")
        probes.setdefault(core, "|".join(alternatives))

    # Standalone acronyms. Two letters are noise ("EU" matches inside words and
    # carries no fact), so only three or more.
    for acronym in re.findall(r"\b([A-Z]{3,5})\b", reference):
        probes.setdefault(acronym, rf"\b{acronym}\b")

    # Drop probes wholly contained in a longer one; they add rows, not signal.
    labels = sorted(probes, key=len, reverse=True)
    for i, label in enumerate(labels):
        if any(label in longer for longer in labels[:i]):
            probes.pop(label, None)
    return probes


def parse_fact_spec(spec: str) -> dict[str, str]:
    """`label=regex; label=regex` from the command line."""
    probes: dict[str, str] = {}
    for clause in spec.split(";"):
        clause = clause.strip()
        if not clause:
            continue
        label, _, pattern = clause.partition("=")
        probes[label.strip()] = pattern.strip() or re.escape(label.strip())
    return probes


def probe_text(text: str, probes: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Literal search. No inference, no gold labels -- just: is it in the string."""
    found: dict[str, dict[str, Any]] = {}
    for label, pattern in probes.items():
        match = re.search(pattern, text, re.I)
        found[label] = {
            "present": bool(match),
            "pattern": pattern,
            "excerpt": text[max(0, match.start() - 60): match.end() + 60].replace("\n", " ")
            if match
            else "",
        }
    return found


def locate_fact_sources(
    probes: dict[str, str], bodies: dict[str, str]
) -> dict[str, list[str]]:
    """Which chunk ids each fact actually appears in."""
    return {
        label: [cid for cid, body in bodies.items() if re.search(pattern, body, re.I)]
        for label, pattern in probes.items()
    }


# --------------------------------------------------------- gold-link grading
def classify_gold_links(
    gold_ids: list[str], bodies: dict[str, str], probes: dict[str, str]
) -> list[dict[str, Any]]:
    """How much of the reference each linked chunk actually supports.

    "5/5 relevant" treats every linked chunk as equal evidence. They are not. A
    chunk that contains the DOJ lawsuit and the €500 million fine is primary
    evidence; a Risk Factors chunk that merely says litigation is a risk is
    indirect, and counting it toward recall inflates the retrieval metric while
    telling the answering model nothing.

      primary    >= 2 required facts
      supporting == 1
      weak       == 0
    """
    out: list[dict[str, Any]] = []
    for cid in gold_ids:
        body = bodies.get(cid, "")
        matched = [
            label for label, pattern in probes.items() if re.search(pattern, body, re.I)
        ]
        if len(matched) >= 2:
            grade_label = "primary"
        elif len(matched) == 1:
            grade_label = "supporting"
        else:
            grade_label = "weak"
        out.append(
            {
                "chunk_id": cid,
                "classification": grade_label,
                "facts_matched": matched,
                "n_facts": len(matched),
                "body_chars": len(body),
                "retrieved": cid in bodies,
            }
        )
    return out


# ------------------------------------------------------------------- autopsy
@dataclass(slots=True)
class Autopsy:
    case_id: str
    git_sha: str
    config: dict[str, Any] = field(default_factory=dict)

    question: str = ""
    reference: str = ""
    answer: str = ""
    passed: bool = False
    judge_rationale: str = ""

    system_prompt: str = ""
    rendered_prompt: str = ""
    context_text: str = ""

    hits: list[dict[str, Any]] = field(default_factory=list)
    gold_ids: list[str] = field(default_factory=list)
    gold_included: list[str] = field(default_factory=list)
    gold_excluded_by_cap: list[str] = field(default_factory=list)
    gold_never_retrieved: list[str] = field(default_factory=list)
    gold_link_quality: list[dict[str, Any]] = field(default_factory=list)

    facts_in_prompt: dict[str, dict[str, Any]] = field(default_factory=dict)
    facts_in_answer: dict[str, dict[str, Any]] = field(default_factory=dict)
    fact_sources: dict[str, list[str]] = field(default_factory=dict)

    context_report: dict[str, Any] = field(default_factory=dict)
    probes_were_derived: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    max_completion_tokens: int = 0
    total_ms: int = 0
    error: str = ""

    def coverage_rows(self) -> list[dict[str, Any]]:
        """required fact | in prompt | source chunk | in answer."""
        rows = []
        for label in self.facts_in_prompt:
            sources = self.fact_sources.get(label, [])
            rows.append(
                {
                    "required_fact": label[:44],
                    "in_prompt": "yes" if self.facts_in_prompt[label]["present"] else "NO",
                    "source_chunks": ",".join(sources[:3]) or "-",
                    "in_answer": "yes"
                    if self.facts_in_answer.get(label, {}).get("present")
                    else "NO",
                }
            )
        return rows


def autopsy(
    case: EvalCase,
    top_n: int,
    use_rerank: bool,
    item_boost: float,
    context_max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    context_packing: str = "greedy-stop",
    strategy: str | None = None,
    mode: str = "hybrid",
    k: int | None = None,
    facts: dict[str, str] | None = None,
    out_path: str | None = None,
) -> Autopsy:
    s = get_settings()
    strategy = strategy or s.default_strategy
    k = k or s.retrieve_k
    probes = facts or derive_fact_probes(case.expected)
    probes_were_derived = facts is None

    report = Autopsy(
        case_id=case.case_id,
        git_sha=git_sha(),
        config={
            "top_n": top_n,
            "use_rerank": use_rerank,
            "item_boost": item_boost,
            "context_max_chars": context_max_chars,
            "context_packing": context_packing,
            "strategy": strategy,
            "mode": mode,
            "k": k,
            "llm_model": s.llm_model,
            "judge_model": s.judge_model,
            "inferred_items": infer_expected_items(case.question),
        },
        question=case.question,
        reference=case.expected,
        gold_ids=list(case.relevant_chunk_ids),
        probes_were_derived=probes_were_derived,
        max_completion_tokens=600,
    )

    trace: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        answer, _retrieval, latency, p_tok, c_tok, result, ctx = answer_question(
            case,
            strategy,
            mode,
            use_rerank,
            k,
            top_n,
            item_boost=item_boost,
            context_max_chars=context_max_chars,
            context_packing=context_packing,
            trace=trace,
        )
    except Exception as exc:  # noqa: BLE001
        report.error = str(exc)[:800]
        report.total_ms = int((time.perf_counter() - started) * 1000)
        return report

    gold = set(case.relevant_chunk_ids)
    included = set(ctx.included_ids)

    report.answer = answer
    report.prompt_tokens = p_tok
    report.completion_tokens = c_tok
    report.total_ms = latency
    report.system_prompt = trace.get("system", "")
    report.rendered_prompt = trace.get("prompt", "")
    report.context_text = trace.get("context", "")
    report.context_report = ctx.as_dict()

    # Every selected hit, in the order the prompt presents them.
    for rank, hit in enumerate(result.hits, start=1):
        report.hits.append(
            {
                "prompt_rank": rank,
                "chunk_id": hit.chunk_id,
                "item": hit.item,
                "relevant": hit.chunk_id in gold,
                "included_in_prompt": hit.chunk_id in included,
                "body_chars": len(hit.body),
                "body_preview": hit.body[:BODY_PREVIEW_CHARS],
            }
        )

    retrieved_bodies = {h.chunk_id: h.body for h in result.candidates}
    report.gold_included = [cid for cid in case.relevant_chunk_ids if cid in included]
    report.gold_excluded_by_cap = [cid for cid in ctx.dropped_ids if cid in gold]
    report.gold_never_retrieved = [
        cid for cid in case.relevant_chunk_ids if cid not in retrieved_bodies
    ]

    # Facts: searched in the literal prompt, never inferred from the gold label.
    report.facts_in_prompt = probe_text(report.rendered_prompt, probes)
    report.facts_in_answer = probe_text(answer, probes)
    report.fact_sources = locate_fact_sources(
        probes, {cid: b for cid, b in retrieved_bodies.items() if cid in included}
    )
    report.gold_link_quality = classify_gold_links(
        case.relevant_chunk_ids, retrieved_bodies, probes
    )

    passed, _score, rationale, _verdict = grade(case, answer)
    report.passed = passed
    report.judge_rationale = rationale

    if out_path:
        save(report, out_path)
    return report


def save(report: Autopsy, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, indent=2)
    return path


def load(path: str) -> Autopsy:
    with open(path, encoding="utf-8") as fh:
        return Autopsy(**json.load(fh))


# ------------------------------------------------------------------- replays
def replay_generation(report: Autopsy, repeats: int = 3) -> list[dict[str, Any]]:
    """Regenerate from the SAVED prompt, with retrieval bypassed entirely.

    This is the clean separation: identical bytes in, so any variation is the
    model, and any failure is the answer prompt rather than anything upstream.
    """
    s = get_settings()
    llm = get_llm()
    out: list[dict[str, Any]] = []
    for i in range(repeats):
        started = time.perf_counter()
        try:
            completion = llm.complete(
                report.rendered_prompt,
                system=report.system_prompt or ANSWER_SYSTEM,
                temperature=0.0,
                max_tokens=report.max_completion_tokens or 600,
                json_schema=ANSWER_SCHEMA,
            )
            try:
                answer = str(parse_json_strict(completion.text).get("answer", "")).strip()
            except Exception:
                answer = completion.text.strip()
            out.append(
                {
                    "repeat": i + 1,
                    "answer": answer,
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                    "hit_token_ceiling": completion.completion_tokens
                    >= (report.max_completion_tokens or 600) - 5,
                    "cost_usd": s.cost_usd(completion.prompt_tokens, completion.completion_tokens),
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                }
            )
        except Exception as exc:  # noqa: BLE001
            out.append({"repeat": i + 1, "error": str(exc)[:400]})
    return out


def replay_judge(
    report: Autopsy,
    answers: list[str] | None = None,
    repeats: int = 3,
    judge_model: str | None = None,
) -> list[dict[str, Any]]:
    """Re-judge saved answers, holding the answer fixed.

    The mirror image of `replay_generation`. Generation is isolated by fixing
    the prompt; the judge is isolated by fixing the answer. If a good answer is
    failed repeatedly, retrieval work cannot move the score, and the judge
    prompt is the thing to change.

    `judge_model` overrides the configured judge for the replay only, so two
    judges can be compared on identical inputs.
    """
    from ..config import reset_settings_cache

    case = EvalCase(
        case_id=report.case_id,
        kind="narrative",
        question=report.question,
        expected=report.reference,
    )
    answers = answers or [report.answer]

    previous = os.environ.get("EDGAR_JUDGE_MODEL")
    if judge_model:
        os.environ["EDGAR_JUDGE_MODEL"] = judge_model
        reset_settings_cache()
    try:
        out: list[dict[str, Any]] = []
        for idx, answer in enumerate(answers, start=1):
            for i in range(repeats):
                try:
                    verdict = judge_narrative(case, answer)
                    out.append(
                        {
                            "answer_index": idx,
                            "repeat": i + 1,
                            "judge_model": judge_model or get_settings().judge_model,
                            "verdict": verdict.verdict,
                            "rationale": verdict.rationale,
                            "latency_ms": verdict.latency_ms,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    out.append({"answer_index": idx, "repeat": i + 1, "error": str(exc)[:400]})
        return out
    finally:
        if judge_model:
            if previous is None:
                os.environ.pop("EDGAR_JUDGE_MODEL", None)
            else:
                os.environ["EDGAR_JUDGE_MODEL"] = previous
            reset_settings_cache()


def judge_prompt_preview(report: Autopsy) -> str:
    """Exactly what the judge is shown. It never sees the source passages."""
    return JUDGE_SYSTEM + "\n\n" + JUDGE_TEMPLATE.format(
        question=report.question, reference=report.reference, candidate=report.answer
    )


def diagnose(report: Autopsy, judge_runs: list[dict[str, Any]] | None = None) -> str:
    """Name the stage at fault, from the measurements rather than a hypothesis."""
    lines: list[str] = []
    n_gold = len(report.gold_ids)
    lines.append(
        f"{len(report.gold_included)}/{n_gold} gold chunks reached the prompt "
        f"({len(report.gold_excluded_by_cap)} cut by the {report.config.get('context_max_chars')}"
        f"-char cap, {len(report.gold_never_retrieved)} never retrieved)."
    )

    missing_from_prompt = [k for k, v in report.facts_in_prompt.items() if not v["present"]]
    missing_from_answer = [
        k
        for k, v in report.facts_in_answer.items()
        if not v["present"] and report.facts_in_prompt.get(k, {}).get("present")
    ]

    if missing_from_prompt:
        lines.append(
            f"RETRIEVAL/PACKING: {len(missing_from_prompt)} required fact(s) are "
            f"absent from the prompt text: {', '.join(missing_from_prompt[:4])}. "
            "The model cannot state what it was not shown."
        )
    elif missing_from_answer:
        lines.append(
            f"GENERATION: every required fact is in the prompt, but "
            f"{len(missing_from_answer)} did not reach the answer "
            f"({', '.join(missing_from_answer[:4])}). Inspect ANSWER_SYSTEM and "
            "the completion token ceiling before touching retrieval."
        )
    elif not report.passed:
        lines.append(
            "JUDGE: every required fact is present in the prompt AND stated in "
            "the answer, and the case still fails. The failure is downstream of "
            "generation. Retrieval work cannot move this number."
        )
    else:
        lines.append("PASS: facts present in prompt and answer, judge agrees.")

    if report.completion_tokens >= (report.max_completion_tokens or 600) - 5:
        lines.append(
            f"The answer hit the {report.max_completion_tokens}-token ceiling, so "
            "length may be truncating content. Raise it before concluding anything else."
        )

    if judge_runs:
        verdicts = [r.get("verdict") for r in judge_runs if "verdict" in r]
        if verdicts and not any(verdicts):
            lines.append(
                f"The judge failed this answer {len(verdicts)}/{len(verdicts)} times. "
                "Read its rationale against the reference text literally -- a judge "
                "that penalises correct detail absent from a summary reference is "
                "measuring the reference, not the answer."
            )

    if report.probes_were_derived and (missing_from_prompt or missing_from_answer):
        lines.append(
            "Probes were derived from the reference automatically, so a NO can "
            "be a wording difference rather than a missing fact. Confirm each NO "
            "against the excerpt before acting on it, or pass --facts."
        )

    weak = [g for g in report.gold_link_quality if g["classification"] == "weak"]
    if weak:
        lines.append(
            f"{len(weak)}/{n_gold} gold links contain none of the required facts "
            f"({', '.join(g['chunk_id'] for g in weak)}). Recall counts them equally "
            "with the chunks that carry the answer, which inflates the retrieval "
            "metric and misdirects tuning."
        )
    return " ".join(lines)
