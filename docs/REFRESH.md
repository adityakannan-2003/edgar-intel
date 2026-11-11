# Scheduled data refresh

Once a week a GitHub Actions job asks SEC EDGAR whether any of the eight
companies has filed a 10-K that the live database lacks. If one has, the job
builds it into a staging copy of the live corpus: sections, chunks, embeddings
and XBRL facts. It checks the figures and the chunks, adds golden-set questions
from the new filing's XBRL facts, and runs an eval gate. Only if everything
passes does it write to the live database, in one transaction. A blocked run
changes nothing, and its report names the filing and the check that failed.

```bash
EDGAR_DB_DSN=<empty staging db> REFRESH_LIVE_DSN=<live db> edgar-intel refresh run [--dry-run] [--tickers AAPL,COST]
REFRESH_LIVE_DSN=<live db> edgar-intel refresh golden --out evalset/golden-refreshed.json   # export the stored golden set
```

Exit codes: 0 means nothing new or published; 1 means a check blocked the
run; 2 means the run failed. Code: `src/edgar_intel/refresh.py`; migration:
`sql/006_refresh_runs.sql`; workflow: `.github/workflows/refresh.yml`; tests:
`tests/test_refresh.py`; proof outputs: `reports/refresh/`.

## How a run works

| step | what happens | writes |
|---|---|---|
| 1. find | For each company the live database serves, read EDGAR's submissions file and keep the 10-Ks filed after the newest one live holds. If there are none, stop and report "nothing new". | none |
| 2. stage | Copy live's corpus tables (companies, filings, sections, chunks with their vectors, xbrl_facts) into an **empty** staging database with `COPY`, then `VACUUM ANALYZE`. The job refuses to stage into a database that already holds filings, so a misconfigured DSN can never point it at live. | staging |
| 3. golden set | Read the stored golden set from live. On the first run there is none, so rebuild the original from the staged copy with `eval build`'s own code and require it to match the reference run's 232 questions and answers exactly (see "The golden set" below). Retrieve passages for every existing question ("before"). | staging |
| 4. ingest | For the new filings only: fetch, run the parser doctor's eight checks, parse into sections, chunk (`section_aware`, 512/64, the `index build` defaults), embed with all-MiniLM-L6-v2, and upsert the company's XBRL facts. | staging |
| 5. check | The data checks below. If any fails, the run is **blocked here** and the paid eval is skipped, because it would spend money grading data already known to be bad. | none |
| 6. eval gate | Retrieve for every existing question again ("after"), then answer the new questions with the shipped pipeline (gpt-4o-mini, hybrid, no rerank, top 8, 20,000-character context: the reference run's config and live `/config`). | staging |
| 7. publish | In one live transaction: lock the corpus tables, check that live's id high-water marks still equal the snapshot's, copy every staging row above them, catch up the sequences, insert the new golden cases and the run's `refresh_runs` row. Commit, or roll back everything. | **live** |

`ingest run` is not reused. It re-parses every filing, which deletes and
re-creates sections. `chunks.section_id` cascades on delete, so that would
delete the embedded index and change every chunk id the golden set's evidence
labels point at. The refresh calls the same building blocks (`upsert_filing`,
`parse_filing`, `replace_sections`, `build_chunks`, `embed_pending`,
`extract_facts`, `upsert_facts`) on new filings only. Existing rows are never
updated: `upsert_facts` is `ON CONFLICT DO NOTHING`, and the publish copies
only rows above the snapshot's high-water marks.

**The publish is the deploy.** The Fly app reads Neon directly and caches
nothing from the corpus, so the new filings are served from the commit
onwards. No Fly release is involved, and `ci.yml`'s deploy job is untouched.

### The checks

Every check names its filing (or its company, or "eval gate") in the report.

| check | catches | source |
|---|---|---|
| text extracted, required items located, offsets on body not ToC, items survive splitting, substantive items have content, MD&A retained figures, risk factors not fused, items in order | a parse that silently lost or fused sections | `ingest doctor`'s `diagnose_html`, unchanged |
| filing parses | markup the parser raises on | new |
| same core XBRL facts as last year | a fact the company reported last year that went missing on the way in (filers differ: CAT tags no `NetIncomeLoss` in any year, so a fixed list would block CAT forever) | new |
| XBRL figures appear in the filing text | **a figure that disagrees with the document**. Each core XBRL fact for the new year must appear verbatim in the filing's own sections at a scale it divides exactly ("416,161" for 416,161,000,000; "499.4" for 499,400,000 shares). Two independent SEC sources, the companyfacts API and the 10-K document, must agree. Round values (924,000,000 shares) have no specific enough form and are reported as not checkable; at least 3 in 4 facts must be checkable. | new |
| chunks embedded | chunks without a vector, or empty chunks | new |
| chunk count in line with last year | a filing that chunked to under half or over twice last year's count | new |
| one value per tag and year; fact periods agree with the filing index; year labels advance with period ends; no fiscal year skipped | the D12/D13 class of fiscal-year defects, for the whole history of each company with a new filing | `ingest verify-facts`' four checks, scoped to one company |
| new questions have evidence in their filing | a new golden case the evidence linker cannot tie to its filing's chunks | new |
| eval gate: existing questions retrieve as before | the new filing changing what an old question retrieves. Search filters by company and year, so a new filing should not enter an old question's candidates, but HNSW post-filtering could (`docs/DEPLOY.md` §2d). This is measured rather than assumed: hit@5 over the 232 questions may not drop more than 0.03, the CI gate's tolerance. | new |
| eval gate: new questions answered | a new filing the pipeline cannot answer from | new |

**Calibration.** Before any refresh ran, the new figure, fact and chunk checks
ran against the 24 filings already in the corpus, all known good. The first
version raised 10 false alarms: CAT's missing net-income tag, and round
values that cannot be found verbatim. After the fix that gave the rules above,
the result was 0 failures in 88 checks: 210 of 219 figures were checkable,
all of them found, and 9 were too round to look for.

**The answer floor** is the reference run's numeric accuracy less two
standard errors at the number of new questions: `ref − 2·√(ref(1−ref)/n)`. At
n = 20 that is 0.8606 − 0.155 = 0.706. With so few questions a three-point
tolerance is meaningless (one case is five points), and a sample of that size
from a system as good as the reference would almost never fall below this
floor. The gate fails closed. With the fake model, or with no reference run in
live, it reports "not run" as a failure.

### The golden set

`eval build` cannot rebuild the golden set once the corpus grows. Narrative
cases take each company's latest fiscal year, so they would silently move to
FY2026 against references written from FY2025. The numeric sample is one
seeded shuffle across every covered year, so one new filing would reshuffle
every company's questions. So the refresh freezes the set and only appends to
it: `golden_cases` in live holds the original 232 plus each published filing's
questions, tagged with its `refresh_run_id`.

On the first run, live has no stored set. The job rebuilds the original from
the staged copy, which works because `eval build` reproduces
`evalset/golden.json` exactly (0 of 232 differ, checked 11 Nov). It then
requires the rebuilt set to match the `(case_id, question, expected)` of every
case the reference run graded, and refuses on any difference. The proof run's
export (`refresh golden`) gave back the 232 original cases identical to
`evalset/golden.json`, in order, followed by the 20 new ones.

New questions per filing follow the original set's density: 7 single-hop
questions (shuffled with a seed derived from ticker and year, so they don't
depend on the rest of the corpus) and 3 year-over-year comparisons. The
comparisons take the prior year as printed in the same filing (D12). They use
the original set's templates, case-id formats and evidence linker.

`/stats/eval` is unchanged: the gate's eval run is recorded in staging and is
never copied to live. Publishing a refresh therefore does not replace the
reference number. Re-measuring the full 252-case set and publishing it is a
separate decision.

## Where it runs, and setting it up

**GitHub Actions, Mondays 11:17 UTC** (`.github/workflows/refresh.yml`), plus
on demand from the Actions tab. A manual run defaults to `dry_run: true`.
Staging is the job's own `pgvector/pgvector:pg16` service container, empty on
every run. Live is Neon. The copy uses `COPY` through psycopg, not `pg_dump`,
because Neon runs Postgres 18 and the runner's `pg_dump` is 16. A blocked or
failed run fails the workflow, so GitHub emails the repository owner. The job
summary is the report, and `reports/refresh/` is uploaded as the
`refresh-report` artifact.

Three repository secrets (Settings → Secrets and variables → Actions → New
repository secret). The job checks they are set before doing anything else:

| secret | value |
|---|---|
| `REFRESH_LIVE_DSN` | the Neon `edgar-intel` connection string, the same value as `REMOTE_DSN` in `~/.config/edgar-intel/deploy.env` and the Fly secret `EDGAR_DB_DSN` |
| `EDGAR_USER_AGENT` | the same as `EDGAR_USER_AGENT` in `.env`: a name and a contact email, which SEC fair access requires |
| `EDGAR_LLM_API_KEY` | an OpenAI key for the answer gate (gpt-4o-mini, about $0.013 per refresh that finds a filing; nothing otherwise) |

The first run that is not a dry run creates `refresh_runs` and `golden_cases`
in Neon (`sql/006_refresh_runs.sql`, `CREATE TABLE IF NOT EXISTS`). Every run
after that adds one `refresh_runs` row. A run that finds a new filing stages
first and then publishes. AAPL's and COST's FY2026 10-Ks are on EDGAR and not
in Neon, so the first scheduled run will publish them.

GitHub disables a public repository's scheduled workflows after 60 days
without a commit. The run summary and the dashboard's "last refresh" show
this as a stale date.

**SEC fair access.** Every request carries `EDGAR_USER_AGENT` (the client
refuses one without an `@`). The client enforces 8 requests a second, under
SEC's 10, and backs off on errors. A week with nothing new costs 8 requests,
one submissions file per company (measured, proof 1). Each new filing adds two
more: its document and the company's companyfacts.

**`refresh_runs`** is the run log the monitoring dashboard reads. Agreed with
that session on 11 Nov: `started_at`, `finished_at`, `status` (`ok` for
nothing new or published, `blocked`, `failed`), `filings_added`, plus
`trigger`, `git_sha`, `new_filings` (jsonb) and `report` (the markdown
report). A publish writes its row in the publish transaction, so an `ok` row
with `filings_added > 0` exists exactly when those filings do. Dry runs write
nothing.

## Proofs, 11 Nov 2026

All three ran at `dfd62aa` from a clean Python 3.11 venv installed the way the
workflow installs (CPU torch, then `.[models]`). They ran against a local
stand-in for Neon, `edgar_refresh_live`, a template copy of `edgar_deploy`:
24 filings, 264 sections, 5,468 chunks, 1,363 facts and 36 eval runs, the
same as the deployed corpus. Staging was the empty `edgar_refresh_stage`.
Each proof records an md5 of every row of every live table before and after
(`reports/refresh/proof*/fingerprint_*.txt`). Nothing was written to Neon.
Spend: $0.0134 for the final proof 3, and $0.040 in all, counting two earlier
proof-3 runs during development.

EDGAR on 11 Nov: two of the eight companies filed a 10-K after the corpus was
built (11 Sep). AAPL FY2026 (0000320193-26-000026, filed 3 Nov) and COST
FY2026 (0000909832-26-000093, filed 7 Oct). So proof 3 uses real new filings,
not a held-back one.

### Proof 2: bad data is blocked (`reports/refresh/proof2/`)

`scripts/refresh_proofs/prove_corrupt.py` corrupts what EDGAR returns, not the
code under test. AAPL's companyfacts get FY2026 revenue multiplied by 10
(478,834,000,000 → 4,788,340,000,000), and COST's 10-K is cut to its first
40%, like a truncated download (1,456,592 → 582,636 bytes). Result:
**BLOCKED**, exit 1, 32 seconds, no model call.

```
## What failed: 3

| filing | check | detail |
|---|---|---|
| AAPL 0000320193-26-000026 FY2026 | XBRL figures appear in the filing text | 10 of 11 found (0 too round to look for); NOT FOUND total revenue $4,788.34 billion (XBRL RevenueFromContractWithCustomerExcludingAssessedTax, looked for 4,788,340,000,000 / 4,788,340,000 / 4,788,340) |
| COST 0000909832-26-000093 FY2026 | XBRL figures appear in the filing text | 7 of 9 found (0 too round to look for); NOT FOUND total liabilities $53.24 billion (XBRL Liabilities, looked for 53,242,000,000 / 53,242,000 / 53,242); NOT FOUND total stockholders' equity $35.80 billion (XBRL StockholdersEquity, looked for 35,803,000,000 / 35,803,000 / 35,803) |
| golden set | new questions have evidence in their filing | 18 of 20 linked; unlinked: num-AAPL-RevenueFromContractWithCustomerExcludingAssessedTax-2026, num-COST-Liabilities-2026 |
```

Live afterwards: every corpus table and `eval_runs` is byte-identical. The
only change is that `refresh_runs` and an empty `golden_cases` exist, with
one `blocked` row whose `report` is the text above.

### Proof 3: a real new 10-K flows end to end (`reports/refresh/proof3/`)

`edgar-intel refresh run`, exactly as the workflow calls it. Result: **OK, 2
filings published**, exit 0, 58 seconds. All 35 checks passed.

- Staged 8 companies, 24 filings, 264 sections, 5,468 chunks and 1,363
  facts. The rebuilt golden set matched the reference run's exam question for
  question.
- AAPL FY2026: 249,436 characters, Items 1 to 9A, 161 chunks (144 last
  year), 11 of 11 core figures found in the text. COST FY2026: 222,925
  characters, 143 chunks (140 last year), 9 of 9 found.
- Golden set: 20 new questions, 20 linked to evidence.
- Existing questions: **0 of 232 changed passages**; hit@5 0.7662 → 0.7662.
- New questions: **18/20 = 0.900**, floor 0.706, $0.0134. Both misses are one
  model error. COST's MD&A says "Cash and cash equivalents and short-term
  investments were $21,301"; the model gave that as cash and cash equivalents,
  which XBRL and the balance sheet put at $20,207. That missed the single-hop
  question and the comparison built on it. This is the "wrong line" class
  from METRICS §5, and the label is right.
- Published in one transaction: 2 filings, 22 sections, 304 chunks, 20
  facts, 252 golden cases (232 original + 20 new). Live became 26 filings,
  286 sections, 5,772 chunks and 1,383 facts. A search for AAPL FY2026 now
  returns FY2026 passages from Items 7 and 8.

### Proof 1: nothing new changes nothing (`reports/refresh/proof1/`)

`edgar-intel refresh run --dry-run`, straight after proof 3. Result: **OK:
nothing new**, exit 0. No staging happened. Every live table is
byte-identical afterwards, `refresh_runs` and `golden_cases` included
(`fingerprint.diff` is empty), and staging is untouched. 8 requests to SEC,
one submissions file per company and no documents.

### The publish is all or nothing

`scripts/refresh_proofs/prove_atomic.py`, on separate throwaway copies:

- **A crash mid-publish.** Every staged row was copied into live, and inside
  the transaction live saw 5,772 chunks. Then the run raised before commit.
  Live was left at 5,468 chunks, byte-identical.
- **Another writer moves live after the snapshot.** The run refused with
  "live changed while the refresh ran (high-water {'filings': 24, …} →
  {'filings': 97, …}); nothing was published, run again". Live was unchanged.

### Predictions, scored

The predictions were committed in `9f46f14` before any proof ran, and are
kept as written.

| # | prediction | outcome |
|---|---|---|
| 1 | proof 2 blocked, exit 1, $0 | ✅ |
| 2 | live corpus and `eval_runs` byte-identical; one `blocked` row; `golden_cases` created empty | ✅ |
| 3 | AAPL fails "XBRL figures appear in the filing text" on revenue; its parse checks pass | ✅ |
| 4 | COST fails "required items located", the figure check and "chunk count" | ❌ **only the figure check failed.** The cut fell after the Item 8 heading but before the balance sheet, so all eight parse checks passed, and 88 chunks against 140 is 0.63×, inside the 0.5× bound. The figure check is what caught a truncated filing. |
| 5 | rebuilt golden set matches the reference exam | ✅ |
| 6 | every data check passes on the real filings | ✅ |
| 7 | about 20 new questions, all linked | ✅ 20, 20 linked |
| 8 | 0 of 232 existing questions change passages | ✅ at `dfd62aa`. ❌ in the first two runs (`647154a`, `8ce2e21`): 93 changed; see below. |
| 9 | about 17/20, floor about 0.71, about $0.02 | ✅ 18/20, floor 0.706, $0.0134 |
| 10 | 2 filings, about 22 sections, about 290 chunks, about 20 facts, 252 golden cases | ✅ 2 / 22 / 304 / 20 / 252 |
| 11 | dry run: nothing new, byte-identical, about 8 requests | ✅ 8 |

### What prediction 8 found: tied passages follow the visibility map

The first two proof-3 runs, at `647154a` and `8ce2e21` (the second's report is
in `reports/refresh/first-run-8ce2e21-vacuum-race/`), both reported **93 of 232 existing questions changed passages**, hit@5 0.7662 →
0.7619. That included CAT questions, and CAT had no new filing. 59 of the 93
were the same eight passages reordered, for example ranks 3 and 4 swapped.
That points to ties.

In those runs, autovacuum had VACUUMed staging's freshly bulk-loaded `chunks`
table four seconds before the "before" pass. So "before" ran on an all-visible
table, and "after" ran with 304 new rows on pages not yet marked visible. The
planner's plan follows the visibility map, and search orders tied scores in
whatever order the plan produces. A controlled reproduction with autovacuum
off (`scripts/refresh_proofs/plan_stability.py`, free) gives exactly the
same numbers (`reports/refresh/plan_stability.txt`):

| `MODE`: staging maintenance between the passes | changed | hit@5 |
|---|---|---|
| `vacuum-then-analyze`: VACUUM ANALYZE after the copy, ANALYZE after the ingest (what autovacuum left in the first runs) | 93 of 232 (59 reordered, 34 a different passage) | 0.7662 → 0.7619 |
| `vacuum-both`: VACUUM ANALYZE after both | 0 of 232 | 0.7662 → 0.7662 |
| `none`: no maintenance at all | 0 of 232 | 0.7619 → 0.7619 |

So the new filings moved no existing question. The refresh now runs `VACUUM
ANALYZE` on staging after the copy and after the ingest (`dfd62aa`), so both
passes run on the same physical state. Each run's JSON keeps every question's
before and after passage ids for diagnosis.

The finding is about retrieval, not the refresh. **Which of several tied
passages a question gets depends on the table's physical state.** The same
232 questions score hit@5 0.7619 or 0.7662 on identical data. The fix
belongs in `retrieval/search.py`: add a deterministic tiebreaker, such as
chunk id, to the lexical and fused orderings. That is shared code, so it is
left as a follow-up rather than changed here.

## Known limitations

- **Two chunkers in one index.** The served index's 5,468 chunks were built
  on 15 Sep, before D9's 512-token ceiling (max `token_est` 588). New filings
  are chunked by today's `section_aware` at 512/64, the only chunker the code
  has. The new questions score 0.900 on them, against 0.8606 for the
  reference, so nothing measured is lost. A re-chunk of everything is a
  separate experiment (`docs/METRICS.md` §1).
- **No backfill and no new companies.** The refresh only adds 10-Ks newer
  than live's newest per company, for companies live already serves. 10-K/A
  amendments are ignored, as `ingest` ignores them.
- **A rejected filing stays rejected until someone looks.** Each week the
  blocked filing is found again and blocked again. It is never published
  without a passing run.
- **The answer gate is small.** 10 questions a filing, and the floor allows
  about 15 points at n = 20. It catches a filing the pipeline cannot read. It
  cannot detect a two-point drift. The full 252-case `eval run` is the
  measurement for that.
- **Untested in Actions until merged.** GitHub runs scheduled and manual
  workflows only from the default branch, so this workflow's first real run
  will be after the merge. Every step it runs was exercised locally, in a
  clean venv installed the same way.
