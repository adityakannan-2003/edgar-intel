"""Typed shapes for the evaluation layer."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

CaseKind = Literal["numeric", "narrative"]


@dataclass(slots=True)
class EvalCase:
    """One question with everything needed to grade an answer to it."""

    case_id: str
    kind: CaseKind
    question: str
    expected: str
    # Numeric cases carry the raw value so grading is a tolerance comparison
    # rather than string matching -- "$383.29 billion" and "383285000000" are
    # the same answer and any string-equality check would mark one wrong.
    expected_value: float | None = None
    unit: str | None = None
    # Chunk ids that genuinely contain the evidence. Populated by the evidence
    # linker; empty means retrieval metrics are not reported for this case.
    relevant_chunk_ids: list[str] = field(default_factory=list)
    ticker: str | None = None
    fiscal_year: int | None = None
    tag: str | None = None
    difficulty: str = "single_hop"      # single_hop | multi_hop | comparative
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CaseResult:
    case_id: str
    kind: CaseKind
    passed: bool | None
    score: float
    answer: str
    expected: str
    retrieval: dict[str, float] = field(default_factory=dict)
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    judge_rationale: str = ""
    error: str = ""
    # The model declined to answer rather than guessing. Not a pass, but a
    # different failure from a wrong figure -- see judge.is_abstention.
    abstained: bool = False

    @property
    def answer_hash(self) -> str:
        return hashlib.blake2b(self.answer.encode("utf-8"), digest_size=12).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunSummary:
    run_key: str
    label: str
    git_sha: str
    n_cases: int
    numeric_accuracy: float
    narrative_pass_rate: float
    overall_score: float
    judge_kappa: float | None
    retrieval: dict[str, float]
    p50_latency_ms: int
    p95_latency_ms: int
    total_cost_usd: float
    cost_per_1k_usd: float
    config: dict[str, Any]
    # Where the numeric failures went. Accuracy alone cannot distinguish "the
    # model is wrong" from "retrieval never found the number", and those have
    # opposite fixes; these two split the failures so the run says which.
    # numeric_accuracy + abstention_rate + hallucination_rate == 1.
    abstention_rate: float = 0.0
    hallucination_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


JUDGE_SCHEMA: dict[str, Any] = {
    "title": "judge_verdict",
    "type": "object",
    "properties": {
        "verdict": {
            "type": "boolean",
            "description": "True if the candidate answer is factually consistent "
            "with the reference and actually answers the question.",
        },
        "rationale": {
            "type": "string",
            "description": "One sentence naming the specific discrepancy, or "
            "confirming the key facts matched.",
        },
    },
    "required": ["verdict", "rationale"],
    "additionalProperties": False,
}

ANSWER_SCHEMA: dict[str, Any] = {
    "title": "grounded_answer",
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Indices of the context passages the answer relies on.",
        },
        "confidence": {"type": "number"},
    },
    "required": ["answer", "citations", "confidence"],
    "additionalProperties": False,
}
