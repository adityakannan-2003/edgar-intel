#!/bin/zsh
# Row count and md5 of every row, per table, of the database at $1.
psql "$1" -At -F' ' <<'SQL'
SELECT 'companies', count(*), md5(string_agg(t::text, '|' ORDER BY cik)) FROM companies t
UNION ALL SELECT 'filings', count(*), md5(string_agg(t::text, '|' ORDER BY id)) FROM filings t
UNION ALL SELECT 'sections', count(*), md5(string_agg(t::text, '|' ORDER BY id)) FROM sections t
UNION ALL SELECT 'chunks', count(*), md5(string_agg(t::text, '|' ORDER BY id)) FROM chunks t
UNION ALL SELECT 'xbrl_facts', count(*), md5(string_agg(t::text, '|' ORDER BY id)) FROM xbrl_facts t
UNION ALL SELECT 'eval_runs', count(*), md5(string_agg(t::text, '|' ORDER BY id)) FROM eval_runs t;
SELECT CASE WHEN to_regclass('golden_cases') IS NULL THEN 'golden_cases absent' END;
SELECT CASE WHEN to_regclass('refresh_runs') IS NULL THEN 'refresh_runs absent' END;
SQL
for t in golden_cases refresh_runs; do psql "$1" -At -F' ' -c "SELECT '$t', count(*), md5(string_agg(t::text, '|' ORDER BY 1)) FROM $t t" 2>/dev/null || true; done
