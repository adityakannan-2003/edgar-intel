# edgar-intel refresh BLOCKED

2026-11-11 20:11 UTC · trigger manual (proof 2: corrupted) · git dfd62aa

Staged a copy of live: 8 companies, 24 filings, 264 sections, 5,468 chunks, 1,363 xbrl_facts.
Golden set: live had none stored, so the original 232 cases were rebuilt from the staged copy and matched the reference run's exam question for question.
**Blocked by the data checks. The eval gate was not run: it would spend money grading data already known to be bad. Nothing was published; the live database is unchanged.**

## New 10-Ks on EDGAR: 2

- AAPL 0000320193-26-000026 FY2026, filed 2026-11-03
- COST 0000909832-26-000093 FY2026, filed 2026-10-07

## What failed: 3

| filing | check | detail |
|---|---|---|
| AAPL 0000320193-26-000026 FY2026 | XBRL figures appear in the filing text | 10 of 11 found (0 too round to look for); NOT FOUND total revenue $4,788.34 billion (XBRL RevenueFromContractWithCustomerExcludingAssessedTax, looked for 4,788,340,000,000 / 4,788,340,000 / 4,788,340) |
| COST 0000909832-26-000093 FY2026 | XBRL figures appear in the filing text | 7 of 9 found (0 too round to look for); NOT FOUND total liabilities $53.24 billion (XBRL Liabilities, looked for 53,242,000,000 / 53,242,000 / 53,242); NOT FOUND total stockholders' equity $35.80 billion (XBRL StockholdersEquity, looked for 35,803,000,000 / 35,803,000 / 35,803) |
| golden set | new questions have evidence in their filing | 18 of 20 linked; unlinked: num-AAPL-RevenueFromContractWithCustomerExcludingAssessedTax-2026, num-COST-Liabilities-2026 |

## Every check: 30 of 33 passed

| filing | check | result | detail |
|---|---|---|---|
| AAPL 0000320193-26-000026 FY2026 | text extracted | ok | 249,436 chars |
| AAPL 0000320193-26-000026 FY2026 | required items located | ok | found 1, 1A, 1B, 2, 3, 5, 7, 7A, 8, 9A |
| AAPL 0000320193-26-000026 FY2026 | offsets on body, not table of contents | ok | 10 headings spanning 83% of the document |
| AAPL 0000320193-26-000026 FY2026 | located items survive splitting | ok | all located items became sections |
| AAPL 0000320193-26-000026 FY2026 | substantive items have content | ok | Items 1, 1A and 7 all have full bodies |
| AAPL 0000320193-26-000026 FY2026 | MD&A retained figures | ok | 177 large numbers, 217 table separators |
| AAPL 0000320193-26-000026 FY2026 | risk factors not fused with financials | ok | 86,552 chars, 19 large numbers |
| AAPL 0000320193-26-000026 FY2026 | items in document order | ok | ascending |
| COST 0000909832-26-000093 FY2026 | text extracted | ok | 137,405 chars |
| COST 0000909832-26-000093 FY2026 | required items located | ok | found 1, 1A, 1B, 2, 3, 5, 7, 7A, 8 |
| COST 0000909832-26-000093 FY2026 | offsets on body, not table of contents | ok | 9 headings spanning 78% of the document |
| COST 0000909832-26-000093 FY2026 | located items survive splitting | ok | all located items became sections |
| COST 0000909832-26-000093 FY2026 | substantive items have content | ok | Items 1, 1A and 7 all have full bodies |
| COST 0000909832-26-000093 FY2026 | MD&A retained figures | ok | 157 large numbers, 181 table separators |
| COST 0000909832-26-000093 FY2026 | risk factors not fused with financials | ok | 44,315 chars, 5 large numbers |
| COST 0000909832-26-000093 FY2026 | items in document order | ok | ascending |
| AAPL 0000320193-26-000026 FY2026 | same core XBRL facts as last year | ok | 11 core facts for FY2026, 11 for FY2025 |
| AAPL 0000320193-26-000026 FY2026 | XBRL figures appear in the filing text | FAIL | 10 of 11 found (0 too round to look for); NOT FOUND total revenue $4,788.34 billion (XBRL RevenueFromContractWithCustomerExcludingAssessedTax, looked for 4,788,340,000,000 / 4,788,340,000 / 4,788,340) |
| AAPL 0000320193-26-000026 FY2026 | chunks embedded | ok | 161 chunks, 161 embedded, 0 empty |
| AAPL 0000320193-26-000026 FY2026 | chunk count in line with last year | ok | 161 vs 144 for FY2025 (1.12x; allowed 0.5x to 2x) |
| COST 0000909832-26-000093 FY2026 | same core XBRL facts as last year | ok | 9 core facts for FY2026, 9 for FY2025 |
| COST 0000909832-26-000093 FY2026 | XBRL figures appear in the filing text | FAIL | 7 of 9 found (0 too round to look for); NOT FOUND total liabilities $53.24 billion (XBRL Liabilities, looked for 53,242,000,000 / 53,242,000 / 53,242); NOT FOUND total stockholders' equity $35.80 billion (XBRL StockholdersEquity, looked for 35,803,000,000 / 35,803,000 / 35,803) |
| COST 0000909832-26-000093 FY2026 | chunks embedded | ok | 88 chunks, 88 embedded, 0 empty |
| COST 0000909832-26-000093 FY2026 | chunk count in line with last year | ok | 88 vs 140 for FY2025 (0.63x; allowed 0.5x to 2x) |
| AAPL XBRL facts, all years | one value per tag and year | ok | none conflict |
| AAPL XBRL facts, all years | fact periods agree with the filing index | ok | all within a week |
| AAPL XBRL facts, all years | year labels advance with period ends | ok | all consistent |
| AAPL XBRL facts, all years | no fiscal year skipped | ok | none skipped |
| COST XBRL facts, all years | one value per tag and year | ok | none conflict |
| COST XBRL facts, all years | fact periods agree with the filing index | ok | all within a week |
| COST XBRL facts, all years | year labels advance with period ends | ok | all consistent |
| COST XBRL facts, all years | no fiscal year skipped | ok | none skipped |
| golden set | new questions have evidence in their filing | FAIL | 18 of 20 linked; unlinked: num-AAPL-RevenueFromContractWithCustomerExcludingAssessedTax-2026, num-COST-Liabilities-2026 |
