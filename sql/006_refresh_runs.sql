-- The scheduled data refresh (docs/REFRESH.md): one row per run, and the
-- golden set it grows.
--
-- `refresh_runs` is what the monitoring dashboard reads for "last refresh".
-- status: 'ok' (nothing new, or new filings validated, gated and published),
-- 'blocked' (a data check or the eval gate stopped the publish; the live
-- corpus is unchanged), 'failed' (the job raised; the live corpus is unchanged).
-- A publish writes its row in the same transaction as the filings, so an 'ok'
-- row with filings_added > 0 exists exactly when those filings do.

CREATE TABLE IF NOT EXISTS refresh_runs (
    id             BIGSERIAL PRIMARY KEY,
    started_at     TIMESTAMPTZ NOT NULL,
    finished_at    TIMESTAMPTZ,
    status         TEXT NOT NULL CHECK (status IN ('ok', 'blocked', 'failed')),
    filings_added  INT NOT NULL DEFAULT 0,
    trigger        TEXT,
    git_sha        TEXT,
    new_filings    JSONB NOT NULL DEFAULT '[]',
    report         TEXT
);

CREATE INDEX IF NOT EXISTS refresh_runs_started_idx ON refresh_runs (started_at DESC);

-- The golden set, frozen. `eval build` cannot regenerate it once the corpus
-- grows: narrative cases take each company's latest fiscal year, and the
-- numeric sample is one seeded shuffle across every covered year, so a new
-- filing would silently rewrite existing questions. The refresh therefore
-- stores the set once (from the corpus the reference run graded) and only
-- appends: each published filing adds its own cases, tagged with its run.
CREATE TABLE IF NOT EXISTS golden_cases (
    case_id        TEXT PRIMARY KEY,
    ordinal        INT NOT NULL,               -- the set's order, kept for exports
    body           JSONB NOT NULL,             -- EvalCase.as_dict()
    refresh_run_id BIGINT REFERENCES refresh_runs (id),  -- NULL: the original set
    added_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
