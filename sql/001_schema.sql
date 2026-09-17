-- Schema for edgar-intel.
--
-- Design note worth being able to defend in an interview: embeddings live in
-- Postgres via pgvector rather than a dedicated vector database. For a corpus
-- of this size (tens of thousands of chunks, far under the ~50M-vector mark
-- where purpose-built engines start winning on index build time and tail
-- latency) pgvector keeps vectors, full text, and the XBRL ground truth in one
-- transactional store. That means a single query can filter by company and
-- fiscal year *and* rank by vector distance, with no cross-store consistency
-- problem and no second system to operate.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------- companies
CREATE TABLE IF NOT EXISTS companies (
    cik          TEXT PRIMARY KEY,           -- zero-padded 10-digit CIK
    ticker       TEXT,
    name         TEXT NOT NULL,
    sic          TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS companies_ticker_idx ON companies (ticker);

-- ----------------------------------------------------------------- filings
CREATE TABLE IF NOT EXISTS filings (
    id             BIGSERIAL PRIMARY KEY,
    cik            TEXT NOT NULL REFERENCES companies (cik) ON DELETE CASCADE,
    accession      TEXT NOT NULL UNIQUE,     -- e.g. 0000320193-23-000106
    form           TEXT NOT NULL,            -- 10-K, 10-Q
    filing_date    DATE NOT NULL,
    period_end     DATE,
    fiscal_year    INT,
    primary_doc    TEXT,
    source_url     TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS filings_cik_year_idx ON filings (cik, fiscal_year);
CREATE INDEX IF NOT EXISTS filings_form_idx     ON filings (form);

-- ---------------------------------------------------------------- sections
-- A filing is split into its numbered Items (Item 1A Risk Factors, Item 7
-- MD&A, ...). Section boundaries matter: the section_aware chunking strategy
-- uses them, and the naive strategies deliberately ignore them so the two can
-- be compared on the same corpus.
CREATE TABLE IF NOT EXISTS sections (
    id           BIGSERIAL PRIMARY KEY,
    filing_id    BIGINT NOT NULL REFERENCES filings (id) ON DELETE CASCADE,
    item         TEXT,                        -- "1A", "7", "7A", ...
    title        TEXT,
    ordinal      INT NOT NULL,
    char_len     INT NOT NULL,
    body         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS sections_filing_idx ON sections (filing_id);

-- ------------------------------------------------------------------ chunks
-- Chunks are strategy-scoped: the same filing is chunked several different
-- ways and each variant is stored under its own `strategy` label, so a
-- retrieval run can be pointed at one strategy and the results compared
-- apples-to-apples on identical source text.
CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    filing_id     BIGINT NOT NULL REFERENCES filings (id) ON DELETE CASCADE,
    section_id    BIGINT REFERENCES sections (id) ON DELETE CASCADE,
    strategy      TEXT NOT NULL,
    ordinal       INT NOT NULL,
    token_est     INT NOT NULL,
    body          TEXT NOT NULL,
    body_tsv      TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', body)) STORED,
    embedding     VECTOR(384),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (filing_id, strategy, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_strategy_idx ON chunks (strategy);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx      ON chunks USING GIN (body_tsv);

-- HNSW over cosine distance. Build this AFTER bulk-loading embeddings; on an
-- empty table it costs nothing, on a loaded one it is the slow step.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ------------------------------------------------------------- xbrl_facts
-- Machine-verifiable ground truth. Every numeric question in the golden set is
-- generated from a row here, which is what stops the evaluation from being an
-- LLM grading another LLM with no external anchor.
CREATE TABLE IF NOT EXISTS xbrl_facts (
    id            BIGSERIAL PRIMARY KEY,
    cik           TEXT NOT NULL REFERENCES companies (cik) ON DELETE CASCADE,
    taxonomy      TEXT NOT NULL,              -- us-gaap, dei
    tag           TEXT NOT NULL,              -- Revenues, NetIncomeLoss, ...
    unit          TEXT NOT NULL,              -- USD, shares
    fiscal_year   INT NOT NULL,
    fiscal_period TEXT NOT NULL,              -- FY, Q1..Q4
    period_start  DATE,
    period_end    DATE,
    value         NUMERIC NOT NULL,
    accession     TEXT,
    form          TEXT,
    UNIQUE (cik, taxonomy, tag, unit, fiscal_year, fiscal_period, period_end)
);

CREATE INDEX IF NOT EXISTS xbrl_lookup_idx ON xbrl_facts (cik, tag, fiscal_year, fiscal_period);

-- -------------------------------------------------------------- eval_runs
CREATE TABLE IF NOT EXISTS eval_runs (
    id             BIGSERIAL PRIMARY KEY,
    run_key        TEXT NOT NULL UNIQUE,
    git_sha        TEXT,
    label          TEXT,
    config         JSONB NOT NULL,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ,
    n_cases        INT NOT NULL DEFAULT 0,
    summary        JSONB
);

CREATE TABLE IF NOT EXISTS eval_results (
    id             BIGSERIAL PRIMARY KEY,
    run_id         BIGINT NOT NULL REFERENCES eval_runs (id) ON DELETE CASCADE,
    case_id        TEXT NOT NULL,
    kind           TEXT NOT NULL,             -- numeric | narrative
    passed         BOOLEAN,
    score          DOUBLE PRECISION,
    retrieval      JSONB,                     -- recall@k, mrr, ndcg, hit ranks
    latency_ms     INT,
    prompt_tokens  INT,
    completion_tokens INT,
    cost_usd       DOUBLE PRECISION,
    answer         TEXT,
    expected       TEXT,
    question       TEXT,
    context_text   TEXT,
    judge_rationale TEXT,
    UNIQUE (run_id, case_id)
);

CREATE INDEX IF NOT EXISTS eval_results_run_idx ON eval_results (run_id);

-- -------------------------------------------------------- judge_calibration
-- Human labels used to calibrate the LLM judge. Cohen's kappa between these
-- and the judge's own verdicts is reported on every run; if it drops, the
-- judge is no longer trustworthy and the narrative scores mean nothing.
CREATE TABLE IF NOT EXISTS judge_calibration (
    id            BIGSERIAL PRIMARY KEY,
    case_id       TEXT NOT NULL,
    answer_hash   TEXT NOT NULL,
    human_label   BOOLEAN NOT NULL,
    labeller      TEXT,
    notes         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (case_id, answer_hash)
);

-- ------------------------------------------------------------ agent_traces
CREATE TABLE IF NOT EXISTS agent_traces (
    id            BIGSERIAL PRIMARY KEY,
    trace_id      TEXT NOT NULL,
    question      TEXT NOT NULL,
    steps         JSONB NOT NULL,
    outcome       TEXT NOT NULL,             -- answered | escalated | exhausted | refused
    total_latency_ms INT,
    total_cost_usd   DOUBLE PRECISION,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_traces_outcome_idx ON agent_traces (outcome);
