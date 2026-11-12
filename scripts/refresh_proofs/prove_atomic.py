"""The publish is all or nothing. Free: no model calls.

A: the publish raises after every row is copied into live, before commit.
B: another writer adds a filing to live after the snapshot; the publish must refuse.
Each compares the live fingerprint (count + md5 per table) before and after.

    EDGAR_DB_DSN=<empty staging> REFRESH_LIVE_DSN=<throwaway live copy> python scripts/refresh_proofs/prove_atomic.py
"""
import hashlib
import os

import psycopg
from psycopg.rows import dict_row

from edgar_intel import refresh
from edgar_intel.ingest.edgar_client import EdgarClient

live = psycopg.connect(os.environ["REFRESH_LIVE_DSN"], row_factory=dict_row, autocommit=True)
stage = psycopg.connect(os.environ["EDGAR_DB_DSN"], row_factory=dict_row, autocommit=True)


def fingerprint():
    out = {}
    for t in (*refresh.CORPUS, "golden_cases", "refresh_runs"):
        rows = live.execute(f"SELECT t::text AS r FROM {t} t ORDER BY 1").fetchall()
        out[t] = (len(rows), hashlib.md5("|".join(r["r"] for r in rows).encode()).hexdigest()[:12])
    return out


refresh._ensure_tables(live)
refresh.stage_copy(live, stage)
hwm = refresh.high_water(stage)
with EdgarClient() as client:
    new = refresh.find_new(live, client)
    refresh.ingest_new(client, new, progress=lambda m: None)
print("staged rows above the snapshot:", {t: stage.execute(f"SELECT COUNT(*) AS n FROM {t} WHERE id > %s", (hwm[t],))
      .fetchone()["n"] for t in refresh.SERIAL})
run = {"started_at": refresh.datetime.now(refresh.UTC), "status": "ok", "filings_added": len(new),
       "trigger": "proof", "git_sha": "", "new_filings": new, "report": None}

# A: crash after the copies, inside the transaction
before = fingerprint()
real = refresh._bump_sequences
def crash(conn):
    copied = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    raise RuntimeError(f"simulated crash mid-publish (live transaction sees {copied} chunks)")
refresh._bump_sequences = crash
try:
    refresh.publish(live, stage, hwm, run, None, [])
except RuntimeError as exc:
    print("A raised:", exc)
refresh._bump_sequences = real
print("A live unchanged:", fingerprint() == before, fingerprint()["chunks"])

# B: a concurrent writer moves live after the snapshot
live.execute("INSERT INTO filings (cik, accession, form, filing_date) VALUES "
             "((SELECT cik FROM companies LIMIT 1), 'proof-concurrent-writer', '10-K', '2026-11-11')")
before = fingerprint()
try:
    refresh.publish(live, stage, hwm, run, None, [])
except refresh.RefreshError as exc:
    print("B refused:", exc)
print("B live unchanged:", fingerprint() == before)
