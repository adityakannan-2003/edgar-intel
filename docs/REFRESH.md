# Scheduled data refresh

(Being written. The predictions below were committed before any proof run.)

## Predictions, written 11 Nov 2026 before any run

EDGAR, checked 11 Nov: two of the eight companies filed a 10-K after the corpus
was built on 11 Sep. AAPL FY2026 (0000320193-26-000026, filed 3 Nov) and COST
FY2026 (0000909832-26-000093, filed 7 Oct). The other six companies' newest
10-Ks are already in the corpus. The proofs run against a local stand-in for
Neon (`edgar_refresh_live`, a template copy of `edgar_deploy`: 24 filings, 264
sections, 5,468 chunks, 1,363 facts, 36 eval runs, the same as the deployed
corpus).

Proof 2, corrupted (AAPL FY2026 revenue ×10 in companyfacts; COST's 10-K cut to its first 40%):

1. Status `blocked`, exit 1. No model call, so $0.
2. Every corpus table and `eval_runs` in live is byte-identical afterwards (md5 per table).
   `refresh_runs` gains one `blocked` row, and `golden_cases` is created empty.
3. AAPL fails "XBRL figures appear in the filing text", naming total revenue at ten
   times its value. AAPL's eight parse checks pass.
4. COST fails "required items located" (Items 7A and 8 are past the cut), "XBRL figures
   appear in the filing text" (the statements are in Item 8), and "chunk count in line
   with last year" (under 0.5×).
5. The original golden set, rebuilt from the staged copy, matches the reference run's
   232 questions exactly.

Proof 3, real filings (AAPL FY2026, COST FY2026):

6. Every data check passes.
7. About 20 new questions: 7 single-hop and 3 year-over-year per filing, all with linked evidence.
8. 0 of the 232 existing questions retrieve different passages (hit@5 unchanged).
9. New-question accuracy about 17 of 20 (0.85), over the floor of about 0.71
   (0.8606 less two standard errors at n=20). Cost about $0.02.
10. Published in one transaction: 2 filings, about 22 sections, about 290 chunks, about
    20 facts, 252 golden cases (232 original + about 20 new). The live corpus becomes
    26 filings and about 5,760 chunks. `/search` for AAPL FY2026 finds FY2026 passages.

Proof 1, dry run after proof 3:

11. "Nothing new", exit 0, no staging. Every table in live, `refresh_runs` and
    `golden_cases` included, is byte-identical afterwards. About 8 requests to SEC
    (one submissions file per company, and no documents).
