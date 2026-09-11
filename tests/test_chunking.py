"""Chunking strategies and filing parsing."""

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
from edgar_intel.ingest.parse import find_item_offsets, html_to_text, split_sections
from edgar_intel.providers.fake import FakeEmbedder


class TestFixed:
    def test_respects_budget(self):
        text = "x" * 5000
        chunks = chunk_fixed(text, target_tokens=100, overlap_tokens=0)
        assert all(len(c) <= 100 * CHARS_PER_TOKEN for c in chunks)

    def test_overlap_repeats_tail(self):
        text = "".join(str(i % 10) for i in range(2000))
        chunks = chunk_fixed(text, target_tokens=100, overlap_tokens=25)
        tail = chunks[0][-25 * CHARS_PER_TOKEN :]
        assert chunks[1].startswith(tail)

    def test_covers_whole_text(self):
        text = "abcdefghij" * 200
        joined = "".join(chunk_fixed(text, target_tokens=50, overlap_tokens=0))
        assert joined == text

    def test_overlap_must_be_smaller_than_size(self):
        with pytest.raises(ValueError):
            chunk_fixed("abc", target_tokens=10, overlap_tokens=10)

    def test_rejects_zero_size(self):
        with pytest.raises(ValueError):
            chunk_fixed("abc", target_tokens=0)


class TestRecursive:
    def test_keeps_short_paragraphs_whole(self):
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird one."
        chunks = chunk_recursive(text, target_tokens=200, overlap_tokens=0)
        assert len(chunks) == 1
        assert "First" in chunks[0] and "Third" in chunks[0]

    def test_splits_when_over_budget(self):
        text = "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(60))
        chunks = chunk_recursive(text, target_tokens=60, overlap_tokens=0)
        assert len(chunks) > 1

    def test_splits_oversized_paragraph_by_sentence(self):
        para = " ".join(f"Sentence number {i} of many." for i in range(80))
        chunks = chunk_recursive(para, target_tokens=40, overlap_tokens=0)
        assert len(chunks) > 1
        assert all(c.strip() for c in chunks)

    def test_no_empty_chunks(self):
        text = "One.\n\n\n\nTwo.\n\n   \n\nThree."
        assert all(c.strip() for c in chunk_recursive(text))


class TestSectionAware:
    def test_never_crosses_section_boundary(self):
        """The property the whole strategy exists to guarantee."""
        sections = [
            {"id": 1, "item": "1A", "title": "Risk Factors", "body": "Risk text. " * 100},
            {"id": 2, "item": "7", "title": "MD&A", "body": "Discussion text. " * 100},
        ]
        chunks = chunk_section_aware(sections, target_tokens=50, overlap_tokens=0)
        for chunk in chunks:
            has_risk = "Risk text" in chunk.body
            has_mda = "Discussion text" in chunk.body
            assert not (has_risk and has_mda)

    def test_prefixes_section_header(self):
        sections = [{"id": 1, "item": "1A", "title": "Risk Factors", "body": "Body text."}]
        chunks = chunk_section_aware(sections, target_tokens=500)
        assert chunks[0].body.startswith("[1A Risk Factors]")

    def test_carries_section_id_for_provenance(self):
        sections = [{"id": 42, "item": "7", "title": "MD&A", "body": "Body. " * 50}]
        assert all(c.section_id == 42 for c in chunk_section_aware(sections, 20))

    def test_ordinals_are_globally_sequential(self):
        sections = [
            {"id": 1, "item": "1", "title": "Business", "body": "A. " * 80},
            {"id": 2, "item": "2", "title": "Properties", "body": "B. " * 80},
        ]
        chunks = chunk_section_aware(sections, target_tokens=20)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))


class TestSemantic:
    def test_groups_related_sentences(self):
        text = (
            "Revenue grew strongly. Revenue growth was driven by volume. "
            "Revenue increased again. "
            "Litigation is pending. Legal proceedings continue. Court dates are set."
        )
        chunks = chunk_semantic(text, FakeEmbedder(dim=64).embed, target_tokens=500)
        assert len(chunks) >= 1
        assert " ".join(chunks).replace("  ", " ").count("Revenue") == 3

    def test_short_text_returns_single_chunk(self):
        assert chunk_semantic("One sentence.", FakeEmbedder().embed) == ["One sentence."]

    def test_empty_text(self):
        assert chunk_semantic("", FakeEmbedder().embed) == []


class TestTokenEstimate:
    def test_never_zero(self):
        assert estimate_tokens("a") >= 1

    def test_scales_with_length(self):
        assert estimate_tokens("x" * 400) == 100


class TestParsing:
    def test_table_cells_stay_on_one_line(self):
        """The bug this prevents is the most expensive one in filing RAG.

        If the label and its figures end up on separate lines, a chunk can hold
        the number without holding what it means -- and retrieval then looks
        broken for reasons that have nothing to do with the model.
        """
        html = """
        <table>
          <tr><td>Total net sales</td><td>383,285</td><td>394,328</td></tr>
        </table>
        """
        text = html_to_text(html)
        line = next(line for line in text.splitlines() if "Total net sales" in line)
        assert "383,285" in line and "394,328" in line

    def test_scripts_and_styles_are_dropped(self):
        html = "<html><body><script>var x=1;</script><p>Real text.</p></body></html>"
        text = html_to_text(html)
        assert "Real text." in text
        assert "var x" not in text

    def test_item_offsets_prefer_the_body_over_the_table_of_contents(self, sample_filing_text):
        """Both the ToC and the section itself name each Item.

        Matching the first occurrence puts every section boundary in the table
        of contents, which silently collapses the whole document into one
        section. Offsets must land past the ToC.
        """
        offsets = dict(find_item_offsets(sample_filing_text))
        toc_end = sample_filing_text.index("Item 1. Business", 20)
        assert offsets["1A"] > toc_end
        assert offsets["7"] > offsets["1A"]

    def test_split_sections_labels_items(self, sample_filing_text):
        sections = split_sections(sample_filing_text, min_chars=50)
        items = [s.item for s in sections]
        assert "1A" in items
        assert "7" in items
        risk = next(s for s in sections if s.item == "1A")
        assert "single-source supplier" in risk.body
        assert "Total net sales" not in risk.body

    def test_document_without_items_returns_one_section(self):
        sections = split_sections("Just some prose with no item headings at all." * 20)
        assert len(sections) == 1
        assert sections[0].item is None
