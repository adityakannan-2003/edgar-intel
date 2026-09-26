"""No strategy may emit a chunk above its target (D9, and the ceiling behind D8).

`index report` on 24 Sep showed every strategy over its own target of 512:

    strategy        avg_tokens  min_tokens  max_tokens
    fixed               511         68          512
    recursive           475         64          576   <- overlap added after the check
    section_aware       473         13          588   <- header added on top of that
    semantic            288          1        31476   <- budget never consulted

`semantic` is the dramatic one: a financial table carries almost no sentence
punctuation, so `_SENT_SPLIT` hands it over as one "sentence", and three code
paths emitted such a block whole. A 31,476-token_est chunk is embedded on its
first 256 word-pieces, under 1% of it, so it is effectively unretrievable by the
dense side for anything it contains.

The overshoots in `recursive` and `section_aware` are small by comparison, but
the target is about to be chosen to fit the embedder's window (D8). A ceiling
that holds "except for the overlap and the header" does not fit anything.

So this file pins one invariant across all four strategies, on text shaped like
the filings that broke it -- tables without punctuation, abbreviation
fragments, headings, a run with no whitespace at all -- rather than on polite
prose.
"""

from __future__ import annotations

import pytest

from edgar_intel.chunking import (
    CHARS_PER_TOKEN,
    chunk_fixed,
    chunk_recursive,
    chunk_section_aware,
    chunk_semantic,
    estimate_tokens,
)
from edgar_intel.chunking.strategies import _bounded_pieces
from edgar_intel.providers.fake import FakeEmbedder


def _table(rows: int) -> str:
    """A financial statement as html_to_text renders it: one row per line,
    pipes between cells, and not a single sentence-ending period."""
    return "\n".join(
        f"Segment {i} net sales | {383_285 + i:,} | {394_328 - i:,} | {352_583 + 7 * i:,}"
        for i in range(rows)
    )


PROSE = (
    "Item 1. Business. The Company designs, manufactures and markets smartphones, "
    "personal computers, tablets, wearables and accessories. It sells a variety of "
    "related services. The Company's fiscal year is the 52- or 53-week period that "
    "ends on the last Saturday of September. No. 5 of the U.S. Treasury notes "
    "matured. The Company depends on component suppliers. Some are single-source. "
)

SHAPES = {
    "prose": PROSE * 30,
    "table": _table(400),
    "prose_table_prose": PROSE * 8 + "\n\n" + _table(250) + "\n\n" + PROSE * 8,
    "no_whitespace": "x" * 9_000,
    "fragments": "Item 1. A. B. C. U.S. No. 5. Inc. " * 40,
}

TARGETS = (64, 190, 512)


def _ceiling(target_tokens: int) -> int:
    return target_tokens * CHARS_PER_TOKEN


def _nonspace(text: str) -> str:
    return "".join(text.split())


def _check_ceiling(chunks: list[str], target: int) -> None:
    assert chunks, "a non-empty input produced no chunks"
    for c in chunks:
        assert c.strip(), "empty chunk"
        assert len(c) <= _ceiling(target), (
            f"{len(c)} chars against a ceiling of {_ceiling(target)} "
            f"({estimate_tokens(c)} token_est against {target})"
        )
        assert estimate_tokens(c) <= target


# ----------------------------------------------------------- the invariant
def _violations(make_chunks) -> list[str]:
    """Every (shape, target) where a chunk is empty or over the ceiling.

    One test per strategy sweeping every shape and target, rather than a
    parametrised cell per combination: this is one invariant, and a failure
    should list every place it breaks rather than inflate the test count.
    """
    found: list[str] = []
    for shape, text in SHAPES.items():
        for target in TARGETS:
            chunks = make_chunks(text, target)
            if not chunks:
                found.append(f"{shape} @ {target}: no chunks from non-empty text")
            for c in chunks:
                if not c.strip():
                    found.append(f"{shape} @ {target}: empty chunk")
                elif len(c) > _ceiling(target) or estimate_tokens(c) > target:
                    found.append(
                        f"{shape} @ {target}: {len(c)} chars / {estimate_tokens(c)} "
                        f"token_est against a ceiling of {_ceiling(target)} / {target}"
                    )
    return found


class TestNoStrategyExceedsItsTarget:
    def test_fixed(self):
        assert not _violations(lambda text, n: chunk_fixed(text, n, n // 8))

    def test_recursive_including_its_overlap(self):
        for divisor in (0, 8, 2):
            assert not _violations(
                lambda text, n, d=divisor: chunk_recursive(text, n, n // d if d else 0)
            ), f"overlap = target // {divisor}"

    def test_section_aware_including_its_header(self):
        def chunks(text: str, n: int) -> list[str]:
            sections = [
                {
                    "id": 7,
                    "item": "7",
                    "title": "Management's Discussion and Analysis of Financial "
                    "Condition and Results of Operations",
                    "body": text,
                },
                {"id": 8, "item": "1A", "title": "Risk Factors", "body": PROSE * 5},
            ]
            return [c.body for c in chunk_section_aware(sections, n, min(64, n // 4))]

        assert not _violations(chunks)

    def test_semantic(self):
        embed = FakeEmbedder(dim=32).embed
        assert not _violations(lambda text, n: chunk_semantic(text, embed, n))


# ------------------------------------------------------------ semantic, D9
class TestSemanticChunkerPaths:
    """The three paths that used to return a block whole."""

    def test_a_table_between_prose_is_cut_to_the_budget(self):
        """The 31,476-token_est chunk, reproduced at small scale."""
        text = PROSE * 4 + "\n\n" + _table(600) + "\n\n" + PROSE * 4
        assert len(_table(600)) > 20 * _ceiling(190)
        chunks = chunk_semantic(text, FakeEmbedder(dim=32).embed, target_tokens=190)
        assert max(len(c) for c in chunks) <= _ceiling(190)

    def test_the_few_sentences_early_return_is_bounded(self):
        """`len(sentences) <= min_sentences` used to return the whole text."""
        one_sentence_table = _table(300)
        chunks = chunk_semantic(one_sentence_table, FakeEmbedder().embed, target_tokens=100)
        assert len(chunks) > 1
        _check_ceiling(chunks, 100)

    def test_the_no_similarities_path_is_bounded(self):
        chunks = chunk_semantic(
            "A single short sentence.", FakeEmbedder().embed, target_tokens=100, min_sentences=0
        )
        assert chunks == ["A single short sentence."]

    def test_an_oversized_sentence_mid_document_is_split_not_emitted_whole(self):
        """The main loop restarted its buffer as `[nxt]` however long `nxt` was."""
        text = "Revenue grew. Margins held. " + "Z" * 5_000 + ". Costs rose. Debt fell."
        chunks = chunk_semantic(text, FakeEmbedder().embed, target_tokens=100)
        _check_ceiling(chunks, 100)

    def test_the_embedder_is_never_asked_to_embed_an_oversized_unit(self):
        seen: list[int] = []
        embedder = FakeEmbedder(dim=16)

        def spy(texts: list[str]) -> list[list[float]]:
            seen.extend(len(t) for t in texts)
            return embedder.embed(texts)

        chunk_semantic(PROSE * 3 + _table(200), spy, target_tokens=128)
        assert seen and max(seen) <= _ceiling(128)

    def test_no_text_is_lost(self):
        """Fragments are glued to a neighbour, never dropped."""
        text = SHAPES["prose_table_prose"]
        chunks = chunk_semantic(text, FakeEmbedder(dim=32).embed, target_tokens=190)
        assert _nonspace(" ".join(chunks)) == _nonspace(text)

    def test_fragments_do_not_become_chunks_of_their_own(self):
        """`min_tokens 1` came from pieces like "Item 1." and "U.S."."""
        text = SHAPES["fragments"] + SHAPES["prose"]
        chunks = chunk_semantic(text, FakeEmbedder(dim=32).embed, target_tokens=190, min_tokens=8)
        assert min(len(c) for c in chunks) >= 8 * CHARS_PER_TOKEN

    def test_text_shorter_than_the_floor_still_comes_back(self):
        assert chunk_semantic("Yes.", FakeEmbedder().embed) == ["Yes."]


# --------------------------------------------------------- recursive overlap
class TestRecursiveOverlapStillWorks:
    def test_overlap_is_carried_when_it_fits(self):
        text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(20))
        chunks = chunk_recursive(text, target_tokens=100, overlap_tokens=10)
        assert len(chunks) > 2
        for prev, nxt in zip(chunks, chunks[1:], strict=False):
            assert nxt[:20] in prev, "the next chunk should open with the previous tail"

    def test_no_text_is_lost_without_overlap(self):
        text = SHAPES["prose_table_prose"]
        chunks = chunk_recursive(text, target_tokens=190, overlap_tokens=0)
        assert _nonspace("".join(chunks)) == _nonspace(text)


# ------------------------------------------------------------ the splitter
class TestBoundedPieces:
    def test_short_text_is_untouched(self):
        assert _bounded_pieces("  short  ", 100) == ["short"]

    def test_prefers_row_boundaries(self):
        """A table cut mid-row separates a figure from its label."""
        pieces = _bounded_pieces(_table(50), 400)
        rows = set(_table(50).splitlines())
        for piece in pieces:
            assert all(line in rows for line in piece.splitlines())

    def test_balanced_so_no_tiny_tail(self):
        pieces = _bounded_pieces("word " * 420, 2_048)  # 2,100 chars
        assert len(pieces) == 2
        assert min(len(p) for p in pieces) > 2_048 // 4

    def test_hard_cut_when_there_is_no_whitespace(self):
        pieces = _bounded_pieces("x" * 1_001, 100)
        assert all(len(p) <= 100 for p in pieces)
        assert "".join(pieces) == "x" * 1_001

    def test_rejects_a_non_positive_budget(self):
        with pytest.raises(ValueError):
            _bounded_pieces("abc", 0)
