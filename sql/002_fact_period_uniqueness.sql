-- One XBRL fact per (company, tag, unit, fiscal year).
--
-- The original constraint included period_end:
--
--   UNIQUE (cik, taxonomy, tag, unit, fiscal_year, fiscal_period, period_end)
--
-- which permitted several rows for one fiscal year, and that is exactly what
-- happened. `companyfacts` stamps every entry with the fy/fp of the *report*
-- it appeared in, not the period it covers, so Caterpillar's FY2025 10-K
-- emitted FY2025, FY2024 and FY2023 revenue all labelled fy=2025. Three rows,
-- three different values, one fiscal year -- and no error, because period_end
-- made each row unique.
--
-- The golden set is generated from this table, so the corruption propagated
-- silently into the ground truth: duplicate case ids with conflicting expected
-- values, and questions whose expected answer belonged to a different year.
--
-- Dropping period_end from the key makes a second value for a year impossible
-- to insert rather than merely unlikely. Rows stored under the old key must be
-- discarded: their fiscal_year column is wrong at the source and cannot be
-- repaired in place.
--
-- `edgar-intel init` re-runs every file in sql/, so this one has to be a no-op
-- once applied. The discard is therefore keyed on the old constraint still
-- being present -- found by its period_end column, not by name, because
-- Postgres truncated the generated name -- and never runs again after the swap.
-- An unconditional TRUNCATE here once wiped the ground truth on every `init`.

DO $$
DECLARE
    old_key TEXT;
BEGIN
    SELECT c.conname INTO old_key
      FROM pg_constraint c
      JOIN pg_attribute a
        ON a.attrelid = c.conrelid AND a.attnum = ANY (c.conkey)
     WHERE c.conrelid = 'xbrl_facts'::regclass
       AND c.contype = 'u'
       AND a.attname = 'period_end'
     LIMIT 1;

    IF old_key IS NOT NULL THEN
        TRUNCATE TABLE xbrl_facts;
        EXECUTE format('ALTER TABLE xbrl_facts DROP CONSTRAINT %I', old_key);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = 'xbrl_facts'::regclass
           AND conname = 'xbrl_facts_period_key'
    ) THEN
        ALTER TABLE xbrl_facts
            ADD CONSTRAINT xbrl_facts_period_key
            UNIQUE (cik, taxonomy, tag, unit, fiscal_year, fiscal_period);
    END IF;
END $$;
