"""Parser diagnostics.

The doctor's job is to notice a bad parse. So each test feeds it a filing that
is broken in one specific way and asserts it says so — a diagnostic that passes
everything is worse than no diagnostic, because it converts a silent failure
into a confident wrong answer.
"""

from __future__ import annotations

from edgar_intel.ingest.doctor import EXPECTED_ITEMS, diagnose_html


def _check(diag, name):
    return next(c for c in diag.checks if c.name == name)


def healthy_filing_html() -> str:
    """A filing shaped the way the parser expects: ToC, then body, with tables."""
    risk = (
        "<p>The Company depends on a limited number of suppliers for critical "
        "components and certain components are available only from single sources. "
        "Disruption at any such supplier could materially affect the Company's "
        "ability to meet customer demand. Competition is intense and characterised "
        "by rapid technological change and frequent product introductions.</p>"
    ) * 12
    mdna_rows = "".join(
        f"<tr><td>Line item {i}</td><td>{380000 + i * 137:,}</td>"
        f"<td>{394000 + i * 211:,}</td></tr>"
        for i in range(30)
    )
    return f"""
    <html><body>
      <p>Table of Contents</p>
      <p>Item 1. Business</p>
      <p>Item 1A. Risk Factors</p>
      <p>Item 7. Management's Discussion and Analysis</p>
      <p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p>
      <p>Item 8. Financial Statements and Supplementary Data</p>

      <p>Item 1. Business</p>
      <p>{"The Company designs and manufactures products sold worldwide. " * 220}</p>

      <p>Item 1A. Risk Factors</p>
      {risk}

      <p>Item 7. Management's Discussion and Analysis</p>
      <table>{mdna_rows}</table>
      <p>{"Net sales decreased during the fiscal year driven by lower volumes. " * 160}</p>

      <p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p>
      <p>{"The Company uses forward contracts to hedge currency exposure. " * 200}</p>

      <p>Item 8. Financial Statements and Supplementary Data</p>
      <p>{"See the consolidated financial statements and accompanying notes. " * 200}</p>
    </body></html>
    """


class TestHealthyFiling:
    def test_passes_every_check(self):
        diag = diagnose_html("TEST", "acc-1", 2023, healthy_filing_html())
        failed = [c.name for c in diag.checks if not c.passed]
        assert not failed, f"unexpected failures: {failed}"
        assert diag.healthy

    def test_finds_the_required_items(self):
        diag = diagnose_html("TEST", "acc-1", 2023, healthy_filing_html())
        assert EXPECTED_ITEMS <= set(diag.items_found)

    def test_reports_corpus_shape(self):
        diag = diagnose_html("TEST", "acc-1", 2023, healthy_filing_html())
        assert diag.text_chars > 20_000
        assert diag.sections >= 5


class TestDetectsBreakage:
    def test_flags_missing_items(self):
        """A filer whose headings the regex does not match."""
        html = healthy_filing_html().replace("Item 1A. Risk Factors", "RISK FACTORS")
        diag = diagnose_html("TEST", "acc-2", 2023, html)
        check = _check(diag, "required items located")
        assert not check.passed
        assert "1A" in check.detail
        assert not diag.healthy

    def test_flags_dropped_tables(self):
        """The damaging failure: MD&A survives but its figures do not.

        Retrieval degrades and it looks like a model problem. This check is the
        reason that day does not get lost.
        """
        html = healthy_filing_html()
        start = html.index("<table>")
        end = html.index("</table>") + len("</table>")
        diag = diagnose_html("TEST", "acc-3", 2023, html[:start] + html[end:])
        check = _check(diag, "MD&A retained figures")
        assert not check.passed
        assert "tables were probably dropped" in check.detail

    def test_flags_short_extraction(self):
        diag = diagnose_html("TEST", "acc-4", 2023, "<html><body><p>Tiny.</p></body></html>")
        assert not _check(diag, "text extracted").passed
        assert not diag.healthy

    def test_missing_mdna_is_reported_not_crashed(self):
        html = healthy_filing_html().replace(
            "Item 7. Management's Discussion and Analysis", "Something Else Entirely", 2
        )
        diag = diagnose_html("TEST", "acc-5", 2023, html)
        check = _check(diag, "MD&A retained figures")
        assert not check.passed
        assert "not found" in check.detail

    def test_error_filing_renders_without_exploding(self):
        from edgar_intel.ingest.doctor import FilingDiagnosis

        diag = FilingDiagnosis("TEST", "-", None, error="connection reset")
        assert not diag.healthy
        assert "connection reset" in diag.render()


class TestVerdict:
    def test_clean_run_says_so(self):
        from edgar_intel.ingest.doctor import _verdict

        diag = diagnose_html("TEST", "acc-1", 2023, healthy_filing_html())
        assert "parsed cleanly" in _verdict([diag])

    def test_broken_run_names_the_fix(self):
        """A verdict that only says 'something failed' has not helped anyone."""
        from edgar_intel.ingest.doctor import _verdict

        html = healthy_filing_html().replace("Item 1A. Risk Factors", "RISK FACTORS")
        diag = diagnose_html("BROKE", "acc-2", 2023, html)
        verdict = _verdict([diag])
        assert "BROKE" in verdict
        assert "ITEM_PATTERNS" in verdict or "parse.py" in verdict
