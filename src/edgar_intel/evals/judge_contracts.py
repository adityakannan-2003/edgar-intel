"""Versioned judge contracts, so the grading rubric is a controlled variable.

The judge prompt had been an unversioned constant. That makes a prompt change
indistinguishable from a model change in the run history: two runs disagree, the
config records `judge_model`, and nothing records what the judge was asked to
do. Naming and versioning the contract makes it a variable you can hold fixed.

`v1` is the contract that failed `nar-AAPL-legal` three times out of three on an
answer that named every proceeding in the reference. It is kept verbatim, not
deleted -- it is the control arm, and a rubric change is only believable against
the thing it replaced.

`v2` fixes what v1 could not do rather than merely softening it:

  * v1 said "Extra correct detail is fine" while showing the judge only the
    reference. It had no way to check whether extra detail was correct, so it
    penalised it. v2 supplies the SOURCE CONTEXT the answer was generated from,
    which is what makes that clause checkable at all.
  * v1's "consistent with the reference" was read as "present in the reference".
    v2 states the three failure modes separately -- contradiction, missing
    required coverage, unsupported extra claims -- so absence from a summary
    reference is no longer a failure condition.

`v3` answers what 24 human labels said about v2, and v2 stays frozen because it
is now the control arm in turn.

Against human judgement on baseline-v5, v2 scored kappa 0.4167: agreement 17/24,
with **7 false positives and 0 false negatives**. Perfectly one-sided -- the
judge is permissive, not unstable. Worse, supplying the source context (the
a9e9c08 fix) moved its pass rate from 50% to 79% and every one of those extra
passes was wrong: with passages to check against, v2 stopped failing extra
detail and still never checked whether the answer was *complete* or at the right
*scope*.

The 7 errors fall in two groups, and v3 adds a rule for each:

  material completeness -- an answer that covers one part of a multi-part
  disclosure and omits the rest (Apple legal and supply, Caterpillar
  competition, Costco supply, NVIDIA legal). Every sentence true, the answer
  still wrong.

  scope fidelity -- an answer that substitutes a narrower scope for the one
  asked about (Caterpillar FX naming Cat Financial as the primary exposure;
  P&G reporting Beauty-segment figures as company-wide performance when the
  reference says overall unit volume was unchanged and organic growth was 1%).

Both groups pass every check v2 makes. That is why the fix is a rubric addition
rather than a wording change, and why v3 ends with the instruction that a
candidate whose every sentence is true can still fail.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JudgeContract:
    name: str
    system: str
    template: str
    wants_source_context: bool = False

    def render(self, question: str, reference: str, candidate: str, source_context: str = "") -> str:
        return self.template.format(
            question=question,
            reference=reference,
            candidate=candidate,
            source_context=source_context or "(not supplied)",
        )


V1 = JudgeContract(
    name="v1",
    system=(
        "You grade answers about SEC filings. You are strict about facts and "
        "indifferent to style. An answer is correct only if every factual claim in "
        "it is consistent with the reference and it actually addresses the "
        "question. Extra correct detail is fine. A hedge that avoids answering is "
        "not correct. Reply with JSON only."
    ),
    template="""QUESTION:
{question}

REFERENCE:
{reference}

CANDIDATE:
{candidate}

Decide whether the candidate answer is factually consistent with the reference
and answers the question. Return {{"verdict": true|false, "rationale": "..."}}.
""",
    wants_source_context=False,
)


V2 = JudgeContract(
    name="v2",
    system=(
        "You are evaluating whether a candidate answer correctly answers the "
        "question, using the supplied reference answer and source context.\n\n"
        "Judge the candidate on three dimensions:\n\n"
        "1. Contradiction. Fail if the candidate makes a factual claim that "
        "contradicts the reference or the supplied source context.\n\n"
        "2. Required coverage. Fail if the candidate omits a material fact "
        "necessary to answer the question that is clearly included in the "
        "reference.\n\n"
        "3. Unsupported material claims. Extra detail is allowed. Do not fail "
        "merely because a detail is absent from the reference. Fail extra detail "
        "only when it is contradicted by the supplied source context or cannot "
        "reasonably be supported by it.\n\n"
        "Important:\n"
        "- The reference is a concise answer key, not an exhaustive list of every "
        "permissible fact.\n"
        "- \"Not mentioned in the reference\" does not mean \"incorrect.\"\n"
        "- Paraphrases and equivalent terminology are acceptable.\n"
        "- Do not require the candidate to reproduce every minor detail or exact "
        "wording from the reference.\n"
        "- A hedge that avoids answering the question is not correct.\n"
        "- Evaluate whether the answer is substantively correct and complete for "
        "the question asked.\n\n"
        "Reply with JSON only."
    ),
    template="""QUESTION
{question}

REFERENCE
{reference}

SOURCE CONTEXT
{source_context}

CANDIDATE ANSWER
{candidate}

Return JSON only:
{{"verdict": true | false, "rationale": "brief explanation"}}
""",
    wants_source_context=True,
)


# v3's system prompt, as named blocks. Composing it this way is what lets v3_1
# be provably "v3 plus exactly two insertions" rather than a retyped string that
# has silently drifted -- the same reason contracts are versioned at all.
_V3_OPENING = (
    "You are evaluating whether a candidate answer correctly answers the "
    "question, using the supplied reference answer and source context.\n\n"
    "Judge the candidate on these dimensions:\n\n"
    "1. Contradiction. Fail if the candidate makes a factual claim that "
    "contradicts the reference or the supplied source context.\n\n"
)

_V3_MATERIAL_COMPLETENESS = (
    "2. MATERIAL COMPLETENESS\n"
    "A candidate must cover the central facts or categories necessary to "
    "answer the question. An answer can be factually correct yet still FAIL "
    "if it covers only one part of a multi-part disclosure while omitting "
    "other material parts identified in the reference.\n\n"
    "Do not treat every reference detail as mandatory. Distinguish supporting "
    "detail from central answer content. But if removing an omitted fact "
    "would materially change the reader's understanding of the company's "
    "disclosure, the omission is material and the answer should FAIL.\n\n"
)

# The v3_1 delta, part one. v3 cleared kappa 0.60 and was blocked anyway: it
# fixed three false positives and created two false negatives, failing P&G FX
# and P&G legal -- correct answers whose references simply carried more
# supporting detail. MATERIAL COMPLETENESS told the judge an incomplete answer
# fails and never told it where completeness stops.
_V3_1_MATERIALITY_BOUNDARY = (
    "MATERIALITY BOUNDARY\n\n"
    "Completeness does not mean exhaustiveness.\n\n"
    "PASS an answer that states the central disclosure correctly even when it "
    "omits supporting details, examples, dates, procedural history, monitoring "
    "methods, secondary quantitative figures, or other facts that would merely "
    "make the answer more complete.\n\n"
    "An omission is material only when it removes a distinct central claim, "
    "category, event, driver, exposure, or relationship needed to answer the "
    "question, or when the omission changes the reader's understanding of the "
    "company's disclosure.\n\n"
    "Before failing for omission, ask:\n\n"
    "\"If the omitted information were added, would it materially change what "
    "the answer says, or would it only add supporting detail?\"\n\n"
    "If it would only add supporting detail, do not fail the answer.\n\n"
    "For references containing many details, do not require every reference "
    "fact. Identify the smallest set of central claims necessary to answer the "
    "question correctly and grade coverage against those claims.\n\n"
)

_V3_SCOPE_AND_SUPPORT = (
    "3. SCOPE FIDELITY\n"
    "The candidate must preserve the scope of the disclosed fact.\n\n"
    "Company-wide results must not be replaced by segment-level results.\n"
    "One business unit's exposure must not be presented as the company's "
    "primary exposure unless the source supports that characterization.\n"
    "Adjacent risks are not substitutes for the specific risk asked about.\n\n"
    "A response containing true source-supported facts can still FAIL when "
    "those facts answer a narrower, different, or adjacent question.\n\n"
    "4. Unsupported material claims. Extra detail is allowed. Do not fail "
    "merely because a detail is absent from the reference. Fail extra detail "
    "only when it is contradicted by the supplied source context or cannot "
    "reasonably be supported by it.\n\n"
    "When comparing REFERENCE and CANDIDATE:\n"
    "1. Identify the central claims required to answer the question.\n"
    "2. Check whether the candidate covers those central claims.\n"
    "3. Check whether company/segment, period, category, and causal scope "
    "match.\n"
    "4. Only then evaluate unsupported or contradictory details.\n\n"
    "Important:\n"
    "- The reference is a concise answer key, not an exhaustive list of every "
    "permissible fact.\n"
    "- \"Not mentioned in the reference\" does not mean \"incorrect.\"\n"
    "- Paraphrases and equivalent terminology are acceptable.\n"
    "- Do not require the candidate to reproduce every minor detail or exact "
    "wording from the reference.\n"
    "- A hedge that avoids answering the question is not correct.\n\n"
    "Do not PASS merely because every sentence in the candidate is true.\n"
    "The candidate must answer the question materially and at the correct "
    "scope.\n\n"
)

# The v3_1 delta, part two: the counterweight to "Do not PASS merely because
# every sentence is true". Without it that instruction has no opposing force and
# the judge drifts strict, which is what the two P&G false negatives were.
_V3_1_DO_NOT_FAIL = (
    "Do not FAIL a substantively correct answer merely because the reference "
    "is more detailed.\n\n"
)

_V3_CLOSING = "Reply with JSON only."


V3 = JudgeContract(
    name="v3",
    system=(
        _V3_OPENING
        + _V3_MATERIAL_COMPLETENESS
        + _V3_SCOPE_AND_SUPPORT
        + _V3_CLOSING
    ),
    template=V2.template,
    wants_source_context=True,
)


V3_1 = JudgeContract(
    name="v3_1",
    system=(
        _V3_OPENING
        + _V3_MATERIAL_COMPLETENESS
        + _V3_1_MATERIALITY_BOUNDARY      # inserted directly after completeness
        + _V3_SCOPE_AND_SUPPORT
        + _V3_1_DO_NOT_FAIL               # inserted beside the verdict instruction
        + _V3_CLOSING
    ),
    template=V2.template,
    wants_source_context=True,
)


CONTRACTS: dict[str, JudgeContract] = {c.name: c for c in (V1, V2, V3, V3_1)}

# The shipped contract. Still v1 until the A/B says otherwise -- a rubric change
# applied before it is measured would make every run before it incomparable and
# every run after it unexplained.
DEFAULT_CONTRACT = "v2"


def get_contract(name: str | None = None) -> JudgeContract:
    key = name or DEFAULT_CONTRACT
    if key not in CONTRACTS:
        raise ValueError(f"unknown judge contract {key!r}; have {sorted(CONTRACTS)}")
    return CONTRACTS[key]
