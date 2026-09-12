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
-- to insert rather than merely unlikely. Existing rows must be discarded: their
-- fiscal_year column is wrong at the source and cannot be repaired in place.

BEGIN;

TRUNCATE TABLE xbrl_facts;

ALTER TABLE xbrl_facts
    DROP CONSTRAINT IF EXISTS xbrl_facts_cik_taxonomy_tag_unit_fiscal_year_fiscal_period_p_key;

DO $$
DECLARE
    conname TEXT;
BEGIN
    SELECT c.conname INTO conname
      FROM pg_constraint c
      JOIN pg_class t ON t.oid = c.conrelid
     WHERE t.relname = 'xbrl_facts'
       AND c.contype = 'u'
     LIMIT 1;
    IF conname IS NOT NULL THEN
        EXECUTE format('ALTER TABLE xbrl_facts DROP CONSTRAINT %I', conname);
    END IF;
END $$;

ALTER TABLE xbrl_facts
    ADD CONSTRAINT xbrl_facts_period_key
    UNIQUE (cik, taxonomy, tag, unit, fiscal_year, fiscal_period);

COMMIT;
