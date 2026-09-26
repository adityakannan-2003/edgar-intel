"""`index build --suffix`: a re-chunk built beside the index it would replace.

`index build` deleted a strategy's chunks before rebuilding them, so measuring
a re-chunk (D8) meant destroying the index it was to be compared with, and
comparing against a number in an old report instead. The chunking function is
chosen by the base strategy; the rows are stored under the label; retrieval
filters on the label alone.
"""

from __future__ import annotations

import pathlib

import pytest

from edgar_intel.retrieval import index as index_mod


class TestLabels:
    @pytest.mark.parametrize(
        "strategy, label, ok",
        [
            ("section_aware", None, True),
            ("section_aware", "section_aware", True),
            ("section_aware", "section_aware_t169", True),
            ("fixed", "section_aware", False),  # would replace another strategy's index
            ("fixed", " fixed_t169", False),
            ("fixed", "   ", False),
        ],
    )
    def test_which_labels_are_allowed(self, strategy, label, ok):
        assert (index_mod.index_label_problem(strategy, label) is None) is ok


class TestBuildingBeside:
    def test_chunks_by_strategy_stores_by_label_and_deletes_only_the_label(self, monkeypatch):
        executed: list = []
        inserted: list = []
        chunked_with: list = []

        monkeypatch.setattr(index_mod.db, "execute", lambda sql, params=None: executed.append(params))
        monkeypatch.setattr(
            index_mod.db, "execute_many", lambda sql, rows: inserted.extend(rows) or len(rows)
        )
        monkeypatch.setattr(index_mod, "_filing_ids", lambda: [1])
        monkeypatch.setattr(index_mod, "get_embedder", lambda: object())
        monkeypatch.setattr(index_mod, "embed_pending", lambda name, *a, **k: 0)

        def build_chunks(filing_id, strategy, target_tokens, overlap):
            chunked_with.append((strategy, target_tokens, overlap))
            return [index_mod.Chunk(body="[7 MD&A] text", ordinal=0, section_id=3)]

        monkeypatch.setattr(index_mod, "build_chunks", build_chunks)

        stats = index_mod.index_strategy(
            "section_aware", 169, 21, label="section_aware_t169"
        )
        assert executed == [("section_aware_t169",)]
        assert chunked_with == [("section_aware", 169, 21)]
        assert {row[2] for row in inserted} == {"section_aware_t169"}
        assert stats.strategy == "section_aware_t169"

    def test_a_forbidden_label_is_refused_before_anything_is_deleted(self, monkeypatch):
        monkeypatch.setattr(
            index_mod.db, "execute", lambda *a, **k: pytest.fail("nothing may be deleted")
        )
        with pytest.raises(ValueError, match="replace it"):
            index_mod.index_strategy("fixed", label="section_aware")


def test_the_cli_exposes_it():
    cli = pathlib.Path("src/edgar_intel/cli.py").read_text()
    assert "suffix: str = typer.Option(" in cli
    assert "label=labels[st]" in cli
