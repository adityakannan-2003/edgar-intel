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


CONTRACTS: dict[str, JudgeContract] = {c.name: c for c in (V1, V2)}

# The shipped contract. Still v1 until the A/B says otherwise -- a rubric change
# applied before it is measured would make every run before it incomparable and
# every run after it unexplained.
DEFAULT_CONTRACT = "v1"


def get_contract(name: str | None = None) -> JudgeContract:
    key = name or DEFAULT_CONTRACT
    if key not in CONTRACTS:
        raise ValueError(f"unknown judge contract {key!r}; have {sorted(CONTRACTS)}")
    return CONTRACTS[key]
