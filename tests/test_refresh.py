"""The refresh job's decisions, without a database or a network (docs/REFRESH.md)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from edgar_intel import refresh
from edgar_intel.evals.schemas import EvalCase
from edgar_intel.refresh import Check, answer_floor, figure_needles, render, retrieval_check


class TestFigureNeedles:
    def test_a_figure_is_looked_for_at_every_scale_it_divides(self):
        assert figure_needles(416_161_000_000, "USD") == ["416,161,000,000", "416,161,000", "416,161"]

    def test_a_corrupted_figure_has_no_needle_the_real_one_matches(self):
        text = "Total net sales | 416,161 | 391,035"
        assert any(n in text for n in figure_needles(416_161_000_000, "USD"))
        assert not any(n in text for n in figure_needles(4_161_610_000_000, "USD"))

    def test_a_balance_rounded_to_tenths_of_a_million_is_looked_for_as_printed(self):
        assert "499.4" in figure_needles(499_400_000, "shares")

    def test_round_values_are_not_checkable_rather_than_missing(self):
        assert figure_needles(924_000_000, "shares") == []
        assert figure_needles(2_000_000_000, "USD") == []

    def test_per_share_values_keep_their_decimals(self):
        assert figure_needles(6.08, "USD/shares") == ["6.08"]


def test_the_answer_floor_is_two_standard_errors_below_the_reference():
    assert answer_floor(0.8606, 20) == pytest.approx(0.8606 - 2 * (0.8606 * 0.1394 / 20) ** 0.5)
    assert round(answer_floor(0.8606, 20), 3) == 0.706
    assert answer_floor(0.8606, 10) < answer_floor(0.8606, 40)


def _case(cid, labels):
    return EvalCase(case_id=cid, kind="numeric", question="q", expected="e", relevant_chunk_ids=labels)


class TestRetrievalCheck:
    cases = [_case("a", ["1"]), _case("b", ["2"])]

    def test_unchanged_passages_pass(self):
        got = {"a": ["1", "9"], "b": ["2"]}
        check = retrieval_check(self.cases, got, dict(got))
        assert check.passed and check.detail.startswith("0 of 2 changed")

    def test_lost_evidence_beyond_tolerance_blocks(self):
        check = retrieval_check(self.cases, {"a": ["1"], "b": ["2"]}, {"a": ["7"], "b": ["2"]})
        assert not check.passed and "changed: a" in check.detail and "1.0000 -> 0.5000" in check.detail


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class _Live:
    def execute(self, *_):
        return _Rows([
            {"cik": "0000320193", "ticker": "AAPL", "name": "Apple Inc.", "latest": date(2025, 10, 31),
             "held": ["0000320193-25-000079"]},
            {"cik": "0000789019", "ticker": "MSFT", "name": "MICROSOFT CORP", "latest": date(2026, 7, 29),
             "held": ["0001193125-26-323660"]},
        ])


class _Client:
    def __init__(self):
        self.asked = []

    def submissions(self, cik):
        self.asked.append(cik)
        filings = {
            "0000320193": [("0000320193-26-000026", "2026-11-03", "2026-09-26"),
                           ("0000320193-25-000079", "2025-10-31", "2025-09-27")],
            "0000789019": [("0001193125-26-323660", "2026-07-29", "2026-06-30")],
        }[cik]
        cols = ("accessionNumber", "filingDate", "reportDate")
        block = {k: [f[i] for f in filings] for i, k in enumerate(cols)}
        block.update(form=["10-K"] * len(filings), primaryDocument=["doc.htm"] * len(filings))
        return {"filings": {"recent": block}}

    def submissions_page(self, name):
        return {}


class TestFindNew:
    def test_only_filings_newer_than_what_live_holds_are_new(self):
        new = refresh.find_new(_Live(), _Client())
        assert [(r["ticker"], r["accession"], r["fiscal_year"]) for r in new] == [
            ("AAPL", "0000320193-26-000026", 2026)
        ]

    def test_tickers_limit_which_companies_are_asked_about(self):
        client = _Client()
        assert refresh.find_new(_Live(), client, ["msft"]) == []
        assert client.asked == ["0000789019"]


def _run(status, **over):
    return {"started_at": datetime(2026, 11, 11, 19, 51, tzinfo=UTC), "status": status, "filings_added": 0,
            "trigger": "schedule", "git_sha": "abc1234", "new_filings": [], **over}


class TestReport:
    def test_a_blocked_run_names_the_filing_and_the_check(self):
        new = [{"ticker": "AAPL", "accession": "0000320193-26-000026", "fiscal_year": 2026,
                "filing_date": "2026-11-03"}]
        checks = [Check("AAPL 0000320193-26-000026 FY2026", "XBRL figures appear in the filing text", False,
                        "NOT FOUND total revenue"), Check("eval gate", "x", True, "got 'net sales | 416,161'")]
        text = render(_run("blocked", new_filings=new), checks, ["Nothing was published."])
        assert text.startswith("# edgar-intel refresh BLOCKED")
        assert "## What failed: 1" in text
        assert "| AAPL 0000320193-26-000026 FY2026 | XBRL figures appear in the filing text | NOT FOUND" in text
        assert "1 of 2 passed" in text
        assert "got 'net sales \\| 416,161' |" in text  # a quoted table row stays inside its cell

    def test_nothing_new_says_so(self):
        assert render(_run("ok"), [], []).startswith("# edgar-intel refresh OK: nothing new")

    def test_a_passing_dry_run_does_not_claim_a_publish(self):
        new = [{"ticker": "COST", "accession": "x", "fiscal_year": 2026, "filing_date": "2026-10-07"}]
        head = render(_run("ok", new_filings=new, dry_run=True), [], []).splitlines()[0]
        assert head == "# edgar-intel refresh OK: 1 filing(s) passed, not published (dry run)"


def test_a_run_that_cannot_start_exits_as_failed(monkeypatch):
    from typer.testing import CliRunner

    from edgar_intel.cli import app

    def unreachable(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(refresh, "run", unreachable)
    result = CliRunner().invoke(app, ["refresh", "run"], env={"REFRESH_LIVE_DSN": "postgresql://x/y"})
    assert result.exit_code == 2 and "connection refused" in result.output


def test_a_pasted_dsn_loses_its_trailing_newline(monkeypatch, tmp_path):
    seen = []

    def connect(dsn, **_):
        seen.append(dsn)
        raise ConnectionError("stop here")

    monkeypatch.setattr(refresh.psycopg, "connect", connect)
    with pytest.raises(ConnectionError):
        refresh.run("postgresql://u:p@host/db?channel_binding=require\n", client=object(), report_dir=str(tmp_path))
    assert seen == ["postgresql://u:p@host/db?channel_binding=require"]


def test_staging_may_not_be_the_live_database(monkeypatch):
    monkeypatch.setenv("EDGAR_DB_DSN", "postgresql://u:p@host/live")
    with pytest.raises(refresh.RefreshError, match="same database"):
        refresh.run("postgresql://u:p@host/live", client=object())


def test_the_cli_exit_code_says_what_happened(monkeypatch):
    from typer.testing import CliRunner

    from edgar_intel.cli import app

    for status, code in (("ok", 0), ("blocked", 1), ("failed", 2)):
        monkeypatch.setattr(refresh, "run", lambda *a, s=status, **k: (s, f"# {s}", "reports/x.md"))
        result = CliRunner().invoke(app, ["refresh", "run"], env={"REFRESH_LIVE_DSN": "postgresql://x/y"})
        assert result.exit_code == code, result.output
        assert f"# {status}" in result.output
        assert "postgresql://x/y" not in result.output
