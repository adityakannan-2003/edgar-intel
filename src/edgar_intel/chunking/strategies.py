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


# ------------------------------------------------------------- the ceiling
# `target_tokens` is a ceiling, not an average. Every strategy below guarantees
# that no chunk it emits is longer than `target_tokens * CHARS_PER_TOKEN`
# characters -- overlap and section headers included -- and
# `tests/test_chunk_bounds.py` asserts it for all four.
#
# It was not always true, and each strategy broke it differently. `semantic`
# returned whole blocks with the budget never consulted, up to 31,476 token_est
# against a target of 512 (D9). `recursive` prepended its overlap tail after the
# budget check, reaching 576. `section_aware` added its header on top of that,
# 588. That mattered little while nobody knew the embedder's window; it matters
# a great deal once the target is chosen to fit it (D8), because a chunk over
# the window is scored by dense retrieval on its opening only.


def _bounded_pieces(text: str, budget: int) -> list[str]:
    """Split `text` into pieces of at most `budget` characters.

    Cuts prefer a line break, then any whitespace, and only then land
    mid-token -- a table row or a figure cut in half is a chunk holding a
    number without its label. The split is balanced rather than greedy: a unit
    of 2,100 characters against a 2,048 budget becomes two pieces of about
    1,050, not 2,048 and 52, so splitting never manufactures the fragment a
    size floor exists to prevent. Every piece of an oversized unit is therefore
    at least about a quarter of the budget.
    """
    text = text.strip()
    if budget <= 0:
        raise ValueError("budget must be positive")
    pieces: list[str] = []
    while len(text) > budget:
        needed = -(-len(text) // budget)       # ceil: pieces still required
        target = -(-len(text) // needed)       # balanced length, <= budget
        cut = _cut_point(text, lo=max(1, target // 2), hi=target)
        head, text = text[:cut].strip(), text[cut:].strip()
        if head:
            pieces.append(head)
    if text:
        pieces.append(text)
    return pieces


def _cut_point(text: str, lo: int, hi: int) -> int:
    """Last line break in text[lo:hi+1], else last whitespace, else `hi`."""
    window = text[lo : hi + 1]
    for sep in ("\n", " ", "\t"):
        at = window.rfind(sep)
        if at != -1:
            return lo + at
    return hi


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
    buf_len = 0  # exact length of " ".join(buf)

    for unit in units:
        ulen = len(unit)
        if buf and buf_len + 1 + ulen > size:
            joined = " ".join(buf)
            chunks.append(joined.strip())
            buf, buf_len = [], 0
            # Carry the overlap forward only as far as the next unit leaves
            # room for it. Prepending it unconditionally after the budget check
            # is how this strategy reached 576 token_est against a 512 target.
            room = min(overlap, size - ulen - 1)
            if room > 0:
                tail = joined[-room:]
                buf, buf_len = [tail], len(tail)
        buf_len = ulen if not buf else buf_len + 1 + ulen
        buf.append(unit)

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
                # Last resort, and a common one: a financial table has almost
                # no sentence punctuation, so the whole table arrives here as
                # one "sentence". Cut it at row boundaries where possible.
                units.extend(_bounded_pieces(sent, size))
    return units


# -------------------------------------------------------------- section_aware
def chunk_section_aware(
    sections: Iterable[dict],
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Chunk within Item boundaries, prefixing each chunk with its section title.

    The header is part of the chunk, so it is paid for out of the same budget:
    the body is chunked to `target_tokens` minus the header's share. Adding it
    afterwards is how this strategy reached 588 token_est against 512.
    """
    size = target_tokens * CHARS_PER_TOKEN
    out: list[Chunk] = []
    ordinal = 0
    for sec in sections:
        title = sec.get("title") or ""
        item = sec.get("item")
        header = f"[{item} {title}] ".strip() if item else (f"[{title}] " if title else "")
        # A header may never take more than half the budget (only reachable
        # with a toy target), so the body always keeps room to say something.
        header = header[: size // 2]
        header_tokens = -(-len(header) // CHARS_PER_TOKEN)
        body_tokens = max(1, target_tokens - header_tokens)
        body_overlap = min(overlap_tokens, body_tokens - 1)
        for piece in chunk_recursive(sec["body"], body_tokens, body_overlap):
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
    min_tokens: int = 8,
) -> list[str]:
    """Split where consecutive-sentence similarity falls into its lowest quartile.

    The percentile threshold is relative to this document rather than an
    absolute cosine value, because similarity distributions differ wildly
    between a Risk Factors section and a footnote table. An absolute threshold
    tuned on one produces nonsense on the other.

    Bounded at both ends (D9). `_SENT_SPLIT` matches almost nothing inside a
    financial-statement table or an exhibit index, so such a block arrives as a
    single "sentence" -- one reached 31,476 token_est, about 125,000
    characters, of which the embedder reads the first 256 word-pieces. Three
    paths used to emit a block like that whole: the two early returns, and the
    main loop restarting its buffer as `[nxt]` however long `nxt` was. Now every
    unit is cut to the budget before grouping, so no path can exceed it.

    At the other end, `min_tokens 1` came from fragments such as "Item 1." or
    "U.S." that the sentence split leaves behind. Those are glued to their
    neighbour rather than dropped: dropping text to fix a size statistic trades
    one silent loss for another.
    """
    budget = target_tokens * CHARS_PER_TOKEN
    # Capped well below the smallest piece `_bounded_pieces` can produce, so
    # gluing and splitting can never fight over the same fragment.
    floor = min(min_tokens * CHARS_PER_TOKEN, budget // 8)
    units = _semantic_units(text, budget, floor)
    if not units:
        return []
    if len(units) <= min_sentences:
        return _pack(units, budget)

    vecs = embed_fn(units)
    sims = [_cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
    if not sims:
        return _pack(units, budget)

    threshold = _percentile(sims, breakpoint_percentile)

    chunks: list[str] = []
    buf = [units[0]]
    buf_len = len(units[0])  # exact length of " ".join(buf)
    for i, sim in enumerate(sims):
        nxt = units[i + 1]
        too_big = buf_len + 1 + len(nxt) > budget
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


def _semantic_units(text: str, budget: int, floor: int) -> list[str]:
    """Sentences, with sub-floor fragments glued forward and every unit <= budget."""
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    glued: list[str] = []
    carry = ""
    for sent in sentences:
        sent = f"{carry} {sent}" if carry else sent
        if len(sent) < floor:
            carry = sent
            continue
        carry = ""
        glued.append(sent)
    if carry:
        # A trailing fragment has nothing after it to join, so it joins the
        # unit before it -- or stands alone if the whole text is that short.
        if glued:
            glued[-1] = f"{glued[-1]} {carry}"
        else:
            glued.append(carry)
    return [piece for sent in glued for piece in _bounded_pieces(sent, budget)]


def _pack(units: list[str], budget: int) -> list[str]:
    """Greedy packing of units already known to fit, for the no-similarity paths."""
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for unit in units:
        if buf and buf_len + 1 + len(unit) > budget:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
        buf_len = len(unit) if not buf else buf_len + 1 + len(unit)
        buf.append(unit)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


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
