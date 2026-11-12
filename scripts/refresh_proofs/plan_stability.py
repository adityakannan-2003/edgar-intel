"""Do existing questions' passages move between the refresh's two retrieval passes? Free: no model calls.

Copies SRC_DSN's corpus into the empty EDGAR_DB_DSN with autovacuum off, so timing cannot decide.
It retrieves for the 232 golden questions, ingests the new 10-Ks, and retrieves again. MODE is the
staging maintenance between the passes (docs/REFRESH.md, "tied passages follow the visibility map"):

  none                  nothing: no statistics, no visibility map
  vacuum-then-analyze   VACUUM ANALYZE after the copy, ANALYZE after the ingest (the first runs' state)
  vacuum-both           VACUUM ANALYZE after both (what refresh.py does)

    MODE=vacuum-both SRC_DSN=.../edgar_deploy EDGAR_DB_DSN=.../empty_db TMP_GOLDEN=/tmp/g.json \
      python scripts/refresh_proofs/plan_stability.py
"""
import os

import psycopg
from psycopg.rows import dict_row

from edgar_intel import db, refresh
from edgar_intel.ingest.edgar_client import EdgarClient

mode = os.environ["MODE"]
live = psycopg.connect(os.environ["SRC_DSN"], row_factory=dict_row, autocommit=True)
stage = psycopg.connect(os.environ["EDGAR_DB_DSN"], row_factory=dict_row, autocommit=True)
db.apply_schema(str(refresh.SQL_DIR))
for t in refresh.CORPUS:
    stage.execute(f"ALTER TABLE {t} SET (autovacuum_enabled = false)")
    refresh._copy(live, stage, t, refresh._columns(live, stage, t))
refresh._bump_sequences(stage)
if mode != "none":
    stage.execute("VACUUM ANALYZE")

old = refresh.bootstrap_golden(live, os.environ["TMP_GOLDEN"])
before = refresh._retrieve(old)
with EdgarClient() as client:
    refresh.ingest_new(client, refresh.find_new(live, client), progress=lambda m: None)
if mode == "vacuum-then-analyze":
    stage.execute("ANALYZE")
elif mode == "vacuum-both":
    stage.execute("VACUUM ANALYZE")
after = refresh._retrieve(old)

changed = [k for k in before if before[k] != after[k]]
reordered = [k for k in changed if set(before[k]) == set(after[k])]
print(f"MODE={mode}: {len(changed)} of {len(old)} changed ({len(reordered)} same passages reordered, "
      f"{len(changed) - len(reordered)} with a different passage); hit@5 "
      f"{refresh._hit_at_5(old, before):.4f} -> {refresh._hit_at_5(old, after):.4f}")
