"""Migrations must survive being re-run.

`edgar-intel init` applies every sql/*.sql file on every run, so a migration that
is only correct the first time is a bug that fires on the second. 002 opened with
an unconditional `TRUNCATE TABLE xbrl_facts`, which meant every `init` silently
deleted the XBRL facts the golden set is generated from.

These are static checks on the SQL text, like the rest of the suite: no database.
"""

from __future__ import annotations

import pathlib
import re

FACT_PERIOD_MIGRATION = pathlib.Path("sql/002_fact_period_uniqueness.sql")

# A PL/pgSQL IF *statement* starts a statement; `DROP ... IF EXISTS` does not.
_IF_OR_END_IF = re.compile(r"(?:\bBEGIN|\bTHEN|\bELSE|\bLOOP|;)\s*IF\b|\bEND\s+IF\b", re.I)


def _code(path: pathlib.Path) -> str:
    return re.sub(r"--[^\n]*", "", path.read_text(encoding="utf-8"))


def _if_depth(body: str, pos: int) -> int:
    depth = 0
    for m in _IF_OR_END_IF.finditer(body, 0, pos):
        depth += -1 if m.group().upper().startswith("END") else 1
    return depth


def _statements(body: str, keyword: str) -> list[int]:
    return [m.start() for m in re.finditer(rf"\b{keyword}\b", body, re.I)]


class TestFactPeriodMigrationIsRerunnable:
    def test_nothing_outside_the_do_block_deletes_rows(self):
        # Outside a $$-quoted body every statement runs on every `init`.
        top_level = _code(FACT_PERIOD_MIGRATION).split("$$")[::2]
        for segment in top_level:
            assert not re.search(r"\b(TRUNCATE|DELETE)\b", segment, re.I), segment

    def test_truncate_only_runs_while_the_old_period_end_key_exists(self):
        bodies = _code(FACT_PERIOD_MIGRATION).split("$$")[1::2]
        truncates = [(b, p) for b in bodies for p in _statements(b, "TRUNCATE")]
        # The discard is still wanted on a database carrying the old key: its
        # rows have the wrong fiscal_year. It just must not run a second time.
        assert truncates, "002 no longer discards rows stored under the old key"
        for body, pos in truncates:
            assert _if_depth(body, pos) > 0, "TRUNCATE is not inside an IF"
            assert "'period_end'" in body, "guard does not look for the old key"

    def test_constraint_changes_are_guarded(self):
        # Unguarded, a re-run would either fail on the existing key or drop it.
        segments = _code(FACT_PERIOD_MIGRATION).split("$$")
        for segment in segments[::2]:
            assert not re.search(r"\bALTER\b", segment, re.I), segment
        for body in segments[1::2]:
            for keyword in ("ALTER", "EXECUTE"):
                for pos in _statements(body, keyword):
                    assert _if_depth(body, pos) > 0, body[pos:pos + 60]
