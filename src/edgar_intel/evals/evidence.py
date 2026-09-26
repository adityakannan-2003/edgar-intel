"""Relevance labels belong to one index build. Check before grading with them.

`relevant_chunk_ids` in the golden set are chunk *ids*, linked against one
strategy (`default_strategy`, section_aware) on the day the set was built.
Two ordinary operations make them wrong without a sound:

- `eval compare` runs the same cases under every strategy. A `fixed` chunk id
  never equals a `section_aware` one, so every other strategy scores
  hit@k = 0 by construction, and the comparison table shows section_aware
  winning retrieval by an infinite margin.
- `index build` deletes a strategy's chunks and re-inserts them. BIGSERIAL
  never reuses an id, so after any re-chunk -- the D8 fix, for instance --
  every label points at a deleted row and the next run reports hit@5 = 0.0
  with no error. That is the void-run defect in a new place: a number that
  describes missing labels, not retrieval.

The labels are derived data -- the linker is deterministic given the index --
so the fix is to re-derive them for the index being graded, with the same
procedure, and to say so in the run config. The human ground truth the linker
searches for (expected values, reference answers, evidence-term overrides) is
untouched.

`index_shape` exists for the same reason. A re-chunk under the same strategy
name leaves `"strategy": "section_aware"` in the config of two runs that graded
different indexes; the chunk count and id range tell them apart.
"""

from __future__ import annotations

import copy
from typing import Any

from .. import db
from .schemas import EvalCase


def label_check(cases: list[EvalCase], strategy: str) -> dict[str, Any]:
    """How many of the cases' relevant chunk ids exist in this strategy's index."""
    ids = sorted({str(cid) for c in cases for cid in c.relevant_chunk_ids})
    record: dict[str, Any] = {
        "strategy": strategy,
        "cases_labelled": sum(1 for c in cases if c.relevant_chunk_ids),
        "labels": len(ids),
        "labels_in_index": 0,
    }
    numeric = [int(i) for i in ids if i.isdigit()]
    if numeric:
        rows = db.query(
            "SELECT id::text AS id FROM chunks WHERE strategy = %s AND id = ANY(%s::bigint[])",
            (strategy, numeric),
        )
        record["labels_in_index"] = len({r["id"] for r in rows} & set(ids))
    return record


def labels_for(
    cases: list[EvalCase], strategy: str
) -> tuple[list[EvalCase], dict[str, Any]]:
    """Cases whose labels are valid for `strategy`'s current index, and a record.

    Labels that all exist in the index are returned untouched. If any are
    missing, every case is re-linked -- on copies, so the caller's golden set is
    never mutated -- and the record says how many were stale and how many cases
    carry labels afterwards, because hit@k over a different population is a
    different number.
    """
    record = label_check(cases, strategy)
    record["stale_labels"] = record["labels"] - record["labels_in_index"]
    record["relinked"] = False
    if record["labels"] == 0 or record["stale_labels"] == 0:
        return cases, record

    from .goldenset import link_evidence

    relinked = link_evidence([copy.deepcopy(c) for c in cases], strategy)
    record["relinked"] = True
    record["cases_labelled_after_relink"] = sum(1 for c in relinked if c.relevant_chunk_ids)
    return relinked, record


def index_shape(strategy: str) -> dict[str, Any]:
    """Which index a run graded against: size, chunk ceiling, and id range."""
    row = db.query_one(
        """
        SELECT COUNT(*)::int                 AS chunks,
               COUNT(embedding)::int         AS embedded,
               COALESCE(MAX(token_est), 0)   AS max_token_est,
               COALESCE(ROUND(AVG(token_est)), 0)::int AS avg_token_est,
               MIN(id)                       AS min_id,
               MAX(id)                       AS max_id
          FROM chunks
         WHERE strategy = %s
        """,
        (strategy,),
    )
    row = dict(row or {})
    return {
        "strategy": strategy,
        "chunks": int(row.get("chunks") or 0),
        "embedded": int(row.get("embedded") or 0),
        "max_token_est": int(row.get("max_token_est") or 0),
        "avg_token_est": int(row.get("avg_token_est") or 0),
        "id_range": [row.get("min_id"), row.get("max_id")],
    }
