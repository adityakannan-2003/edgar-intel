"""Chunking strategies.

Four of them, implemented to be compared rather than to be believed. The whole
point of having more than one is that the evaluation harness can run identical
questions against each and report recall@k per strategy -- which turns "we use
recursive chunking with 512 tokens and 50 overlap" from a guess copied off a
blog post into a measured decision you can defend.

What generally happens on financial filings, and why:

  fixed        Splits on a character budget with no regard for structure.
               Cheap, and it cuts tables in half. A number ends up in one
               chunk and its row label in the next, so retrieval finds a
               passage full of digits that answers nothing.

  recursive    Splits on paragraph, then sentence, then character. Respects
               prose, still ignores Item boundaries, so a chunk can span the
               end of Risk Factors and the start of MD&A -- two passages with
               opposite meaning fused into one embedding.

  section_aware Never crosses an Item boundary and prefixes every chunk with
               its section title. The prefix matters more than it looks: it
               puts "Risk Factors" into the embedded text, so a query about
               risks has lexical and semantic purchase on the right chunks.

  semantic     Groups adjacent sentences by embedding similarity and cuts
               where similarity drops. Best topical coherence, most expensive
               to build (one embedding call per sentence), and the gain over
               section_aware on structured filings is usually small. Measure
               before paying for it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

# ~4 characters per token is the usual rule of thumb for English. Good enough
# for budgeting; anything that needs exactness should tokenise properly.
CHARS_PER_TOKEN = 4

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_PARA_SPLIT = re.compile(r"\n\s*\n")


@dataclass(slots=True)
class Chunk:
    body: str
    ordinal: int
    section_id: int | None = None
    item: str | None = None

    @property
    def token_est(self) -> int:
        return max(1, len(self.body) // CHARS_PER_TOKEN)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


# --------------------------------------------------------------------- fixed
def chunk_fixed(text: str, target_tokens: int = 512, overlap_tokens: int = 64) -> list[str]:
    size = target_tokens * CHARS_PER_TOKEN
    overlap = overlap_tokens * CHARS_PER_TOKEN
    if size <= 0:
        raise ValueError("target_tokens must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than the chunk size")

    out: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        start = end - overlap
    return out


# ----------------------------------------------------------------- recursive
def chunk_recursive(text: str, target_tokens: int = 512, overlap_tokens: int = 64) -> list[str]:
    """Pack paragraphs, then sentences, then characters, up to the budget."""
    size = target_tokens * CHARS_PER_TOKEN
    overlap = overlap_tokens * CHARS_PER_TOKEN

    units = _split_to_units(text, size)
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for unit in units:
        ulen = len(unit)
        if buf and buf_len + ulen > size:
            chunks.append(" ".join(buf).strip())
            if overlap > 0:
                tail = " ".join(buf)[-overlap:]
                buf = [tail]
                buf_len = len(tail)
            else:
                buf, buf_len = [], 0
        buf.append(unit)
        buf_len += ulen + 1

    if buf:
        tail = " ".join(buf).strip()
        if tail:
            chunks.append(tail)
    return [c for c in chunks if c]


def _split_to_units(text: str, size: int) -> list[str]:
    units: list[str] = []
    for para in _PARA_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
            continue
        for sent in _SENT_SPLIT.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= size:
                units.append(sent)
            else:
                units.extend(sent[i : i + size] for i in range(0, len(sent), size))
    return units


# -------------------------------------------------------------- section_aware
def chunk_section_aware(
    sections: Iterable[dict],
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Chunk within Item boundaries, prefixing each chunk with its section title."""
    out: list[Chunk] = []
    ordinal = 0
    for sec in sections:
        title = sec.get("title") or ""
        item = sec.get("item")
        header = f"[{item} {title}] ".strip() if item else (f"[{title}] " if title else "")
        for piece in chunk_recursive(sec["body"], target_tokens, overlap_tokens):
            out.append(
                Chunk(
                    body=f"{header}{piece}".strip(),
                    ordinal=ordinal,
                    section_id=sec.get("id"),
                    item=item,
                )
            )
            ordinal += 1
    return out


# ------------------------------------------------------------------ semantic
def chunk_semantic(
    text: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    target_tokens: int = 512,
    breakpoint_percentile: float = 0.25,
    min_sentences: int = 2,
) -> list[str]:
    """Split where consecutive-sentence similarity falls into its lowest quartile.

    The percentile threshold is relative to this document rather than an
    absolute cosine value, because similarity distributions differ wildly
    between a Risk Factors section and a footnote table. An absolute threshold
    tuned on one produces nonsense on the other.
    """
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sentences) <= min_sentences:
        return [text.strip()] if text.strip() else []

    vecs = embed_fn(sentences)
    sims = [_cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
    if not sims:
        return [text.strip()]

    threshold = _percentile(sims, breakpoint_percentile)
    budget = target_tokens * CHARS_PER_TOKEN

    chunks: list[str] = []
    buf = [sentences[0]]
    buf_len = len(sentences[0])
    for i, sim in enumerate(sims):
        nxt = sentences[i + 1]
        too_big = buf_len + len(nxt) > budget
        topic_shift = sim <= threshold and len(buf) >= min_sentences
        if too_big or topic_shift:
            chunks.append(" ".join(buf).strip())
            buf, buf_len = [nxt], len(nxt)
        else:
            buf.append(nxt)
            buf_len += len(nxt) + 1
    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if c]


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(q * (len(ordered) - 1))
    return ordered[idx]


STRATEGIES = ("fixed", "recursive", "section_aware", "semantic")
