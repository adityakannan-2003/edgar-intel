-- The prior fiscal year's value for the same tag and unit, as printed in the
-- same filing as the row's own value.
--
-- Each row keeps the earliest filing's figure for its year, which is right for
-- "what was X in FY2024" and wrong for "how did X change from FY2024 to
-- FY2025": the two years then come from two filings, and the later one may
-- have restated the earlier year. NVIDIA's 10-for-1 split in June 2024 did
-- exactly that. Its FY2024 10-K reported diluted EPS of $11.93; its FY2025
-- 10-K printed FY2024 as $1.19. Comparing $11.93 with FY2025's $2.94 gave
-- "decreased 75.4%" for a rise of 147% (D12 in docs/METRICS.md).
--
-- Year-over-year comparisons read this column instead. NULL means the filing
-- printed no prior year for the tag, and the comparison falls back to the
-- prior year's own row. Rows ingested before this column existed are NULL
-- everywhere until facts are re-ingested.

ALTER TABLE xbrl_facts
    ADD COLUMN IF NOT EXISTS prior_year_value NUMERIC;
