"""Tied scores are ordered by chunk id, never by the plan Postgres happens to pick.

On identical data the golden set scored hybrid hit@5 0.7619 before a VACUUM of
`chunks` and 0.7662 after it, because tied passages came back in whatever order
the plan produced (docs/METRICS.md §1, "Tied passages").
"""

from __future__ import annotations

import contextlib
import importlib
import re

from edgar_intel.retrieval.search import Hit, reciprocal_rank_fusion

search_mod = importlib.import_module("edgar_intel.retrieval.search")


def hit(chunk_id: str, source: str) -> Hit:
    return Hit(chunk_id=chunk_id, body="", filing_id=1, score=1.0, source=source, ranks={source: 1})


class _Conn:
    def __init__(self, sent: list[str]) -> None:
        self.sent = sent

    def execute(self, sql, params=None):
        self.sent.append(" ".join(sql.split()))
        return self

    def fetchall(self):
        return []


def _transactions(monkeypatch, fn, filters=None) -> list[list[str]]:
    """Statements sent, grouped by the pooled connection (one transaction) that sent them."""
    txns: list[list[str]] = []

    def connection():
        txns.append([])
        return contextlib.nullcontext(_Conn(txns[-1]))

    monkeypatch.setattr(search_mod.db, "connection", connection)
    monkeypatch.setattr(search_mod.db, "query", lambda sql, params: txns.append([" ".join(sql.split())]) or [])
    fn("total revenue fiscal 2024", **({"ticker": "AAPL", "fiscal_year": 2024} if filters is None else filters))
    return txns


def _sql_of(monkeypatch, fn) -> str:
    return _transactions(monkeypatch, fn)[-1][-1]


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
    # `+ 0`: an iterative scan yields rows slightly out of order, and on plain
    # `distance` the planner trusts the inner order (Incremental Sort).
    assert sql.endswith("ORDER BY distance + 0, chunk_id::bigint LIMIT %s")


def test_dense_scans_past_ef_search_in_the_same_transaction(monkeypatch):
    # HNSW stops after ef_search (40) candidates and the filters apply after
    # it: a year-only search got 15 rows, an unfiltered one 0-43, not k=50.
    # SET LOCAL only lasts if it runs on the query's own connection.
    for filters in ({"fiscal_year": 2024}, {}, {"ticker": "AAPL", "fiscal_year": 2024}):
        [txn] = _transactions(monkeypatch, search_mod.dense_search, filters)
        assert txn[0] == "SET LOCAL hnsw.iterative_scan = relaxed_order"
        assert len(txn) == 2 and txn[1].startswith("SELECT * FROM (")
