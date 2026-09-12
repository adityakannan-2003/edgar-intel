"""Grading: numeric comparison, and an LLM judge that has to prove itself.

The numeric path needs no model at all. The narrative path uses an LLM judge,
and the important part of this module is not the judge -- it is the
calibration. An uncalibrated LLM judge is a number generator. Reporting its
pass rate as "quality" without knowing whether it agrees with a human is the
most common unforced error in LLM evaluation.

So: label a sample of answers by hand, store them, and report Cohen's kappa
between the human labels and the judge's verdicts on every run. If kappa falls
below the threshold, the narrative score is flagged as untrustworthy rather
than quietly reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .. import db
from ..config import get_settings
from ..providers import get_llm
from ..providers.openai_compat import parse_json_strict
from ..retrieval.metrics import cohens_kappa
from .schemas import JUDGE_SCHEMA, CaseResult, EvalCase

# Below this, the judge is not tracking human judgement well enough for its
# pass rate to mean anything. 0.6 is the conventional "substantial agreement"
# floor; it is a convention, not a law, and is configurable for a reason.
KAPPA_FLOOR = 0.60

_NUM_IN_TEXT = re.compile(r"-?\$?\s?\d[\d,]*\.?\d*\s?(?:billion|million|thousand|bn|mm|k|%)?", re.I)
_SCALES = {
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
    "million": 1_000_000,
    "mm": 1_000_000,
    "thousand": 1_000,
    "k": 1_000,
}

# Every way a question's own fiscal year leaks into the answer text. These are
# stripped before parsing, because the year is the *subject* of the question and
# never the answer to it.
_YEAR_LABEL = re.compile(
    r"\b(?:FY\s?|fiscal\s+(?:year\s+)?|calendar\s+year\s+|in\s+|for\s+|ended?\s+"
    r"(?:\w+\s+\d{1,2},?\s+)?)(19|20)\d{2}\b",
    re.I,
)
_DATE_LIKE = re.compile(r"\b(19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/](19|20)\d{2}\b")


def _looks_like_a_bare_year(raw: str) -> bool:
    """A four-digit integer in year range, carrying no numeric signal at all.

    "$2,025 million" is a figure. "2025" on its own, in a sentence about fiscal
    2025, is the question repeating itself.
    """
    if "$" in raw or "%" in raw or "," in raw or "." in raw:
        return False
    if any(w in raw.lower() for w in _SCALES):
        return False
    digits = raw.strip().lstrip("-")
    return digits.isdigit() and len(digits) == 4 and 1900 <= int(digits) <= 2100


def _score_candidate(raw: str) -> int:
    """How much this looks like a reported figure rather than incidental digits."""
    lowered = raw.lower()
    if any(w in lowered for w in _SCALES) or "%" in raw:
        return 3
    if "$" in raw:
        return 2
    if "," in raw or "." in raw:
        return 1
    return 0


def parse_number(text: str) -> float | None:
    """Pull the reported figure out of free text, honouring magnitude words.

    "$383.29 billion", "383,285" (in millions, as filings print it) and
    "383285000000" all have to reduce to the same value or numeric grading
    produces false negatives that look like model failures.

    Taking the *first* number is the trap, and it cost this project a whole
    baseline run. "The R&D expense for Caterpillar in FY2025 was $2,148 million"
    parses to 2025 -- the fiscal year in the question, echoed back in the
    answer. Fifteen of twenty graded cases failed that way, every one of them
    reported as a wrong figure rather than as a broken parser.

    So: strip year labels, discard bare four-digit years, and prefer the
    candidate carrying an actual numeric signal -- a currency symbol, a scale
    word, a separator -- over incidental digits. Ties go to the first match.
    """
    cleaned_text = _DATE_LIKE.sub(" ", _YEAR_LABEL.sub(" ", text))

    best: tuple[int, float] | None = None
    for match in _NUM_IN_TEXT.finditer(cleaned_text):
        raw = match.group(0).strip()
        if _looks_like_a_bare_year(raw):
            continue
        value = _to_float(raw)
        if value is None:
            continue
        score = _score_candidate(raw)
        if best is None or score > best[0]:
            best = (score, value)
    return best[1] if best else None


def _to_float(raw: str) -> float | None:
    cleaned = raw.replace("$", "").replace(",", "").replace("%", "").strip()
    scale = 1
    for word, mult in _SCALES.items():
        if cleaned.lower().endswith(word):
            cleaned = cleaned[: -len(word)].strip()
            scale = mult
            break
    try:
        return float(cleaned) * scale
    except ValueError:
        return None


# An answer that declines to answer. The system prompt explicitly asks for this
# when the context lacks the evidence, so it is the behaviour we designed for --
# and it must never be scored as if the model invented a figure.
_ABSTENTION = re.compile(
    r"\b(?:does not (?:provide|contain|include|specify|mention)"
    r"|do not (?:provide|contain|include|specify|mention)"
    r"|is not (?:provided|available|specified|mentioned|included)"
    r"|no (?:information|data|figure|mention)"
    r"|not (?:stated|disclosed|available|specified) in the (?:context|passages?|filing)"
    r"|cannot (?:be )?(?:determine|determined|answer|be answered)"
    r"|unable to (?:determine|answer))\b",
    re.I,
)


def is_abstention(answer: str) -> bool:
    """Did the model decline rather than guess?

    Worth separating from a wrong answer, because the two have opposite fixes
    and opposite meanings. A wrong figure is a generation failure -- the model
    had evidence and misread it, or invented one. An abstention is a *retrieval*
    failure surfacing correctly: the model was handed passages that did not
    contain the answer and said so, which is exactly what the system prompt
    asks for.

    Collapsing them into one "incorrect" bucket is how a project reports 25%
    accuracy and draws the wrong conclusion from it. Of the first twenty graded
    cases, eight failures were abstentions. The generator was behaving; the
    retriever was not finding the number.
    """
    return bool(_ABSTENTION.search(answer))


_DECREASE = re.compile(r"\b(decreas|declin|fell|fall|drop|down|lower|reduc|contract)", re.I)
_INCREASE = re.compile(r"\b(increas|rose|rise|grew|grow|up|higher|gain|expand)", re.I)


def _apply_direction(magnitude: float, answer: str, expected: float) -> float:
    """Recover the sign of a percentage change from the surrounding words.

    If the answer already carries an explicit sign, that wins. Otherwise the
    direction word decides. When neither is present the magnitude is taken at
    face value, which will correctly fail against a negative expectation rather
    than guessing in the candidate's favour.
    """
    if magnitude < 0:
        return magnitude
    if _DECREASE.search(answer) and not _INCREASE.search(answer):
        return -abs(magnitude)
    if _INCREASE.search(answer):
        return abs(magnitude)
    return magnitude


def grade_numeric(
    case: EvalCase, answer: str, tolerance: float | None = None
) -> tuple[bool, float, str]:
    """Compare a numeric answer to the XBRL ground truth within a relative tolerance.

    Scale tolerance is the subtle part: filings present figures "in millions",
    so a model that answers "383,285" when the fact is 383,285,000,000 is
    correct in substance and wrong by a factor of a million. We accept a match
    at any of the common reporting scales and say so in the rationale, rather
    than either failing a right answer or silently accepting a wrong magnitude.
    """
    s = get_settings()
    tol = tolerance if tolerance is not None else s.eval_numeric_tolerance

    if case.expected_value is None:
        return False, 0.0, "case has no expected_value; cannot grade numerically"

    if is_abstention(answer):
        return False, 0.0, "ABSTAINED: the model said the context lacked the answer"

    got = parse_number(answer)
    if got is None:
        return False, 0.0, "no number found in the answer"

    expected = case.expected_value

    # Percentages (year-over-year cases) compare on absolute difference; a 0.5%
    # relative tolerance on a value of 0.3% would be absurdly tight.
    #
    # Direction is the subtle part. A model answering "decreased 3.0%" has
    # stated a negative change, but the minus sign lives in the word, not the
    # number. Comparing the bare figure against an expected -2.8 would mark a
    # correct answer wrong -- and worse, would mark "increased 2.8%" correct
    # when it is the opposite of the truth.
    if case.unit == "percent":
        signed = _apply_direction(got, answer, expected)
        ok = abs(signed - expected) <= 0.5
        return (
            ok,
            1.0 if ok else 0.0,
            f"expected {expected:+.2f}%, read {signed:+.2f}% from the answer",
        )

    for scale, note in ((1, "exact"), (1_000, "thousands"), (1_000_000, "millions")):
        candidate = got * scale
        if expected == 0:
            if candidate == 0:
                return True, 1.0, "both zero"
            continue
        if abs(candidate - expected) / abs(expected) <= tol:
            return True, 1.0, f"match at {note} scale"

    return False, 0.0, f"expected {expected:,.2f}, got {got:,.2f} (no scale matched)"


JUDGE_SYSTEM = (
    "You grade answers about SEC filings. You are strict about facts and "
    "indifferent to style. An answer is correct only if every factual claim in "
    "it is consistent with the reference and it actually addresses the "
    "question. Extra correct detail is fine. A hedge that avoids answering is "
    "not correct. Reply with JSON only."
)

JUDGE_TEMPLATE = """QUESTION:
{question}

REFERENCE:
{reference}

CANDIDATE:
{candidate}

Decide whether the candidate answer is factually consistent with the reference
and answers the question. Return {{"verdict": true|false, "rationale": "..."}}.
"""


@dataclass(slots=True)
class JudgeVerdict:
    verdict: bool
    rationale: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0


def judge_narrative(case: EvalCase, answer: str) -> JudgeVerdict:
    s = get_settings()
    llm = get_llm()
    prompt = JUDGE_TEMPLATE.format(
        question=case.question, reference=case.expected, candidate=answer
    )
    completion = llm.complete(
        prompt,
        system=JUDGE_SYSTEM,
        model=s.judge_model,
        temperature=0.0,
        max_tokens=300,
        json_schema=JUDGE_SCHEMA,
    )
    try:
        payload = parse_json_strict(completion.text)
        verdict = bool(payload.get("verdict", False))
        rationale = str(payload.get("rationale", ""))[:500]
    except Exception as exc:
        verdict, rationale = False, f"judge output unparseable: {exc}"

    return JudgeVerdict(
        verdict=verdict,
        rationale=rationale,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        latency_ms=completion.latency_ms,
    )


# --------------------------------------------------------------- calibration
def record_human_label(case_id: str, answer: str, label: bool, labeller: str = "me",
                       notes: str = "") -> None:
    answer_hash = _hash_answer(answer)
    db.execute(
        """
        INSERT INTO judge_calibration (case_id, answer_hash, human_label, labeller, notes)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (case_id, answer_hash) DO UPDATE
           SET human_label = EXCLUDED.human_label,
               labeller    = EXCLUDED.labeller,
               notes       = EXCLUDED.notes
        """,
        (case_id, answer_hash, label, labeller, notes),
    )


def calibration_pairs(run_id: int) -> list[tuple[bool, bool]]:
    """(human, judge) pairs for every judged answer that has a human label.

    Matched on (case_id, answer_hash) rather than case_id alone: a human label
    applies to the *specific answer* that was labelled. If the system's answer
    to that case has since changed, the old label says nothing about the new
    answer, and silently reusing it would inflate kappa.

    The hash is computed here rather than in SQL so the database needs no
    pgcrypto extension.
    """
    labels = {
        (r["case_id"], r["answer_hash"]): bool(r["human_label"])
        for r in db.query("SELECT case_id, answer_hash, human_label FROM judge_calibration")
    }
    if not labels:
        return []

    rows = db.query(
        """
        SELECT case_id, answer, passed
          FROM eval_results
         WHERE run_id = %s AND kind = 'narrative' AND passed IS NOT NULL
        """,
        (run_id,),
    )
    pairs: list[tuple[bool, bool]] = []
    for row in rows:
        key = (row["case_id"], _hash_answer(row["answer"] or ""))
        if key in labels:
            pairs.append((labels[key], bool(row["passed"])))
    return pairs


def _hash_answer(answer: str) -> str:
    import hashlib

    return hashlib.blake2b(answer.encode("utf-8"), digest_size=12).hexdigest()


def compute_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa between human and judge, or None if too few labels.

    Twenty is a floor, not a target. Below it the estimate is too noisy to act
    on; forty or more makes it reasonably stable.
    """
    if len(pairs) < 20:
        return None
    human = [p[0] for p in pairs]
    judge = [p[1] for p in pairs]
    return cohens_kappa(human, judge)


def kappa_verdict(kappa: float | None, floor: float = KAPPA_FLOOR) -> str:
    if kappa is None:
        return "uncalibrated: fewer than 20 human labels, narrative scores are unverified"
    if kappa >= 0.80:
        return f"kappa={kappa:.2f}: almost perfect agreement with human labels"
    if kappa >= floor:
        return f"kappa={kappa:.2f}: substantial agreement, narrative scores usable"
    return (
        f"kappa={kappa:.2f}: BELOW the {floor:.2f} floor -- the judge is not tracking "
        "human judgement; do not report narrative pass rate as a quality metric"
    )


def sample_for_labelling(run_id: int, n: int = 40) -> list[dict]:
    """Pick answers to hand-label.

    Sampled across both judge verdicts rather than at random, because the
    interesting disagreements cluster where the judge said no. An all-pass
    sample cannot detect a judge that never says no.
    """
    return db.query(
        """
        (SELECT case_id, answer, expected, passed, judge_rationale
           FROM eval_results
          WHERE run_id = %s AND kind = 'narrative' AND passed = true
          ORDER BY random() LIMIT %s)
        UNION ALL
        (SELECT case_id, answer, expected, passed, judge_rationale
           FROM eval_results
          WHERE run_id = %s AND kind = 'narrative' AND passed = false
          ORDER BY random() LIMIT %s)
        """,
        (run_id, n // 2, run_id, n // 2),
    )


def grade(case: EvalCase, answer: str) -> tuple[bool, float, str, JudgeVerdict | None]:
    """Route a case to the right grader."""
    if case.kind == "numeric":
        passed, score, rationale = grade_numeric(case, answer)
        return passed, score, rationale, None
    verdict = judge_narrative(case, answer)
    return verdict.verdict, 1.0 if verdict.verdict else 0.0, verdict.rationale, verdict


def build_result(case: EvalCase, answer: str, retrieval: dict, latency_ms: int,
                 prompt_tokens: int, completion_tokens: int) -> CaseResult:
    s = get_settings()
    passed, score, rationale, verdict = grade(case, answer)
    abstained = case.kind == "numeric" and is_abstention(answer)
    p_tok = prompt_tokens + (verdict.prompt_tokens if verdict else 0)
    c_tok = completion_tokens + (verdict.completion_tokens if verdict else 0)
    return CaseResult(
        case_id=case.case_id,
        kind=case.kind,
        passed=passed,
        score=score,
        answer=answer,
        expected=case.expected,
        retrieval=retrieval,
        latency_ms=latency_ms,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        cost_usd=s.cost_usd(p_tok, c_tok),
        judge_rationale=rationale,
        abstained=abstained,
    )
