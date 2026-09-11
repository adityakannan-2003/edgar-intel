"""Guardrails: input screening, output validation, and loop detection.

Guardrails get talked about as a safety feature. Two thirds of what they
actually do here is keep the agent from wasting money and producing garbage:

  * Input screening rejects out-of-scope questions before any model call. An
    agent asked for stock advice should decline in one cheap step, not after
    eight tool calls.
  * Output validation checks that cited passage ids were actually returned by a
    tool during this run. A citation the agent invented is the single clearest
    hallucination signal available, and it is checkable in code rather than by
    another model.
  * Loop detection catches the most common and expensive agent failure: calling
    the same tool with the same arguments over and over because the result was
    not what it wanted. Without this, a stuck agent burns the entire step
    budget re-asking one question.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# Questions this system is not for. Declining fast is both correct and cheap.
OUT_OF_SCOPE = [
    (re.compile(r"\b(should i|would you) (buy|sell|short|invest)", re.I),
     "This system reports what filings say. It does not give investment advice."),
    (re.compile(r"\b(price target|stock price|share price today|market cap right now)\b", re.I),
     "This system only has filing data, which is historical and does not include market prices."),
    (re.compile(r"\b(predict|forecast|will .*(stock|share).*(rise|fall|go up|go down))\b", re.I),
     "This system does not forecast. It can report what management disclosed about outlook."),
]

# Prompt-injection patterns. Filing text is untrusted input: it is fetched from
# a third party and fed into a model's context, so text inside a document that
# reads like an instruction has to be treated as data.
INJECTION_MARKERS = [
    re.compile(r"ignore (all|any|the) (previous|prior|above) instructions", re.I),
    re.compile(r"\bsystem prompt\b", re.I),
    re.compile(r"\bdisregard your (rules|instructions|guidelines)\b", re.I),
    re.compile(r"</?(system|assistant)>", re.I),
]


@dataclass(slots=True)
class ScreenResult:
    allowed: bool
    reason: str = ""


def screen_input(question: str) -> ScreenResult:
    q = question.strip()
    if len(q) < 3:
        return ScreenResult(False, "Question is empty.")
    if len(q) > 2000:
        return ScreenResult(False, "Question is too long; please narrow it.")
    for pattern, message in OUT_OF_SCOPE:
        if pattern.search(q):
            return ScreenResult(False, message)
    return ScreenResult(True)


def sanitise_tool_output(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate tool output and neutralise instruction-like spans.

    Neutralising means annotating, not deleting: the passage may be genuinely
    relevant, and silently removing text from evidence is its own failure mode.
    Marking it lets the model see the content while knowing it is quoted
    document text rather than an instruction addressed to it.
    """
    flagged = False
    for pattern in INJECTION_MARKERS:
        if pattern.search(text):
            flagged = True
            text = pattern.sub(lambda m: f"[quoted document text: {m.group(0)}]", text)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n[truncated at {max_chars} characters]"
    return text, flagged


@dataclass(slots=True)
class LoopDetector:
    """Flags repeated identical tool calls.

    `repeat_limit=2` means a call may be retried once -- transient failures are
    real -- but a third identical call is a loop and gets refused with a message
    telling the agent to change approach or escalate.
    """

    repeat_limit: int = 2
    seen: dict[str, int] = field(default_factory=dict)

    def record(self, tool: str, args: dict) -> bool:
        key = self._key(tool, args)
        self.seen[key] = self.seen.get(key, 0) + 1
        return self.seen[key] > self.repeat_limit

    def count(self, tool: str, args: dict) -> int:
        return self.seen.get(self._key(tool, args), 0)

    @staticmethod
    def _key(tool: str, args: dict) -> str:
        payload = f"{tool}:{sorted((str(k), str(v)) for k, v in args.items())}"
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


def validate_citations(answer: str, cited: list[str], available: set[str]) -> tuple[bool, str]:
    """Every cited id must have come from a tool result in this run."""
    if not cited:
        return True, ""
    invented = [c for c in cited if c not in available]
    if invented:
        return False, (
            f"Citations {invented} were never returned by any tool in this run. "
            "Cite only passage ids you actually retrieved."
        )
    return True, ""


NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")

# Below this, a number is prose rather than a financial figure: counts,
# percentages, segment tallies, list indices.
_MATERIALITY_FLOOR = 1000
# Four-digit values in this range are almost always years, which appear in
# narrative answers all the time without coming from a tool.
_YEAR_RANGE = range(1900, 2101)


def _numbers_in(text: str) -> set[float]:
    out: set[float] = set()
    for raw in NUMBER.findall(text):
        try:
            out.add(float(raw.replace(",", "").replace("$", "")))
        except ValueError:
            continue
    return out


def validate_numeric_grounding(answer: str, tool_outputs: list[str]) -> tuple[bool, str]:
    """Check that figures in the answer appeared in some tool output.

    Compares numerically rather than by substring. Substring matching looks
    simpler and is wrong in both directions: "$1,234,567." fails to match
    "1234567" because of punctuation, while "383" spuriously matches inside
    "383285". Parsing both sides to floats removes the whole class of problem.

    Deliberately lenient about what counts as a figure: values under a
    materiality floor, and anything that looks like a year, are skipped. The
    purpose is to catch a fabricated revenue number, not to argue about the
    number 3.
    """
    haystack = _numbers_in(" ".join(tool_outputs))

    unseen: list[str] = []
    for raw in NUMBER.findall(answer):
        try:
            value = float(raw.replace(",", "").replace("$", ""))
        except ValueError:
            continue
        if abs(value) < _MATERIALITY_FLOOR:
            continue
        if value.is_integer() and int(value) in _YEAR_RANGE:
            continue
        # Filings restate the same figure at different scales ("in millions").
        # An answer quoting a scaled version of a retrieved number is grounded.
        if any(
            abs(value - candidate * scale) <= max(1.0, abs(candidate * scale) * 1e-6)
            for candidate in haystack
            for scale in (1, 1_000, 1_000_000)
        ):
            continue
        unseen.append(raw)

    if unseen:
        return False, f"Figures not present in any tool output: {unseen[:5]}"
    return True, ""
