"""Tied scores are ordered by chunk id, never by the plan Postgres happens to pick.

On identical data the golden set scored hybrid hit@5 0.7619 before a VACUUM of
`chunks` and 0.7662 after it, because tied passages came back in whatever order
the plan produced (docs/METRICS.md §1, "Tied passages").
"""

from __future__ import annotations

import importlib
import re

from edgar_intel.retrieval.search import Hit, reciprocal_rank_fusion

search_mod = importlib.import_module("edgar_intel.retrieval.search")


def hit(chunk_id: str, source: str) -> Hit:
    return Hit(chunk_id=chunk_id, body="", filing_id=1, score=1.0, source=source, ranks={source: 1})


def _sql_of(monkeypatch, fn) -> str:
    sent: list[str] = []
    monkeypatch.setattr(search_mod.db, "query", lambda sql, params: sent.append(sql) or [])
    fn("total revenue fiscal 2024", ticker="AAPL", fiscal_year=2024)
    return " ".join(sent[0].split())


def test_fusion_orders_tied_scores_by_chunk_id_whichever_list_comes_first():
    # Rank 1 in one list each: equal RRF scores.
    a = reciprocal_rank_fusion([[hit("62001", "dense")], [hit("57400", "lexical")]])
    b = reciprocal_rank_fusion([[hit("57400", "dense")], [hit("62001", "lexical")]])
    assert [h.chunk_id for h in a] == [h.chunk_id for h in b] == ["57400", "62001"]


def test_fusion_still_ranks_by_score_first():
    fused = reciprocal_rank_fusion([[hit("9", "dense"), hit("1", "dense")], [hit("9", "lexical")]])
    assert [h.chunk_id for h in fused] == ["9", "1"]


def test_lexical_breaks_rank_ties_by_id(monkeypatch):
    sql = _sql_of(monkeypatch, search_mod.lexical_search)
    assert re.search(r"ORDER BY ts_rank_cd\(.*\) DESC, c\.id LIMIT", sql)


def test_dense_breaks_distance_ties_by_id_and_keeps_the_index_usable(monkeypatch):
    sql = _sql_of(monkeypatch, search_mod.dense_search)
    # The inner ORDER BY must stay bare: pgvector's HNSW index cannot serve
    # `ORDER BY distance, c.id`, and every unfiltered search would go exact.
    assert "ORDER BY distance FETCH FIRST %s ROWS WITH TIES" in sql
    assert sql.endswith("ORDER BY distance, chunk_id::bigint LIMIT %s")
