"""Proof 2: one corrupted figure and one corrupted filing, through the real refresh.

The corruption is applied to what EDGAR returns, not to the code under test:
  AAPL  companyfacts: FY2026 revenue (the 10-K's own fact) multiplied by 10.
  COST  10-K document: cut off after its first 40% (a truncated download).

    EDGAR_DB_DSN=<empty staging> REFRESH_LIVE_DSN=<live stand-in> \
      python scripts/refresh_proofs/prove_corrupt.py reports/refresh/proof2
"""
import os
import sys

from edgar_intel import refresh
from edgar_intel.ingest.edgar_client import EdgarClient

AAPL, COST = "0000320193", "0000909832"


class CorruptingClient(EdgarClient):
    def company_facts(self, cik):
        cf = super().company_facts(cik)
        if cik.endswith(AAPL[-6:]):
            tag = cf["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]
            for e in tag["units"]["USD"]:
                if e.get("end") == "2026-09-26" and e.get("form") == "10-K" and e.get("start", "") < "2025-10-01":
                    print(f"corrupting AAPL revenue {e['val']:,} -> {e['val'] * 10:,} (accn {e['accn']})")
                    e["val"] *= 10
        return cf

    def filing_document(self, cik, accession, primary_doc):
        html = super().filing_document(cik, accession, primary_doc)
        if cik.endswith(COST[-6:]):
            print(f"truncating COST {accession}: {len(html):,} -> {int(len(html) * 0.4):,} bytes")
            html = html[: int(len(html) * 0.4)]
        return html


status, report, path = refresh.run(os.environ["REFRESH_LIVE_DSN"], trigger="manual (proof 2: corrupted)",
                                   report_dir=sys.argv[1], client=CorruptingClient())
print(report)
print("status:", status, "report:", path)
