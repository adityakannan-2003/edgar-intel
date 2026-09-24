"""Expanding the narrative set from 24 to 40 without corrupting the 24.

The set is 8 companies x 5 topics = 40. Only 3 topics per company were ever
generated, so 16 reference answers do not exist yet. Two things can go wrong in
that expansion and neither would fail loudly:

  The seeded shuffle drifts. Case ids come out of `Random(7)` advanced across
  companies. If that ordering changes, `nar-AAPL-competition` can become a
  different (company, topic) cell than the one a human labelled, and 24 hand
  labels silently become labels of something else.

  A question override holds an answer. `NARRATIVE_QUESTION_OVERRIDES` replaces
  the *question*; the UNH entry held a paragraph of reference prose about
  privacy breaches while that case's reference answer is about provider
  dependence. The golden set on disk predated the override, so nothing failed --
  the next rebuild would have graded answers to one question against the
  reference for another.
"""

from __future__ import annotations

import pytest

from edgar_intel.evals import narrative_sources as ns
from edgar_intel.evals.goldenset import (
    NARRATIVE_QUESTION_OVERRIDES,
    NARRATIVE_REFERENCES,
    NARRATIVE_SEEDS,
    check_question_overrides,
)

TOPICS = {"supply-concentration", "competition", "fx-exposure", "revenue-drivers", "legal"}

COMPANIES = [
    {"cik": "0000320193", "ticker": "AAPL", "name": "Apple Inc."},
    {"cik": "0000018230", "ticker": "CAT", "name": "CATERPILLAR INC"},
    {"cik": "0000909832", "ticker": "COST", "name": "COSTCO WHOLESALE CORP /NEW"},
    {"cik": "0000200406", "ticker": "JNJ", "name": "JOHNSON & JOHNSON"},
    {"cik": "0000789019", "ticker": "MSFT", "name": "MICROSOFT CORP"},
    {"cik": "0001045810", "ticker": "NVDA", "name": "NVIDIA CORP"},
    {"cik": "0000080424", "ticker": "PG", "name": "PROCTER & GAMBLE Co"},
    {"cik": "0000731766", "ticker": "UNH", "name": "UNITEDHEALTH GROUP INC"},
]

EXPECTED_MISSING = {
    "nar-AAPL-competition",
    "nar-AAPL-fx-exposure",
    "nar-CAT-supply-concentration",
    "nar-CAT-legal",
    "nar-COST-competition",
    "nar-COST-legal",
    "nar-JNJ-revenue-drivers",
    "nar-JNJ-legal",
    "nar-MSFT-supply-concentration",
    "nar-MSFT-revenue-drivers",
    "nar-NVDA-supply-concentration",
    "nar-NVDA-competition",
    "nar-PG-supply-concentration",
    "nar-PG-competition",
    "nar-UNH-fx-exposure",
    "nar-UNH-revenue-drivers",
}


@pytest.fixture
def fake_db(monkeypatch):
    """The eight ingested companies, and one fiscal year each. No database."""

    def query(sql, params=None):
        if "FROM companies" in sql:
            return list(COMPANIES)
        return []

    def query_one(sql, params=None):
        if "MAX(fiscal_year)" in sql:
            return {"fy": 2025}
        return None

    monkeypatch.setattr("edgar_intel.db.query", query)
    monkeypatch.setattr("edgar_intel.db.query_one", query_one)


class TestThePlanReproducesTheExistingSet:
    def test_three_per_company_yields_exactly_the_24_already_written(self, fake_db):
        """The shuffle is the thing that must not drift."""
        plan = ns.narrative_plan(max_per_company=3)
        assert len(plan) == 24
        assert {p.case_id for p in plan} == set(NARRATIVE_REFERENCES)

    def test_every_planned_case_at_three_has_a_reference(self, fake_db):
        assert ns.missing_references(ns.narrative_plan(max_per_company=3)) == []

    def test_the_plan_needs_no_references_to_exist(self, fake_db):
        """The dumper runs before any of the 16 answers is written."""
        plan = ns.narrative_plan(max_per_company=5)
        assert len(plan) == 40
        assert any(not p.has_reference for p in plan)


class TestTheFortyCaseGrid:
    def test_five_topics_are_seeded(self):
        assert {s["slug"] for s in NARRATIVE_SEEDS} == TOPICS

    def test_eight_companies_times_five_topics(self, fake_db):
        plan = ns.narrative_plan(max_per_company=5)
        assert len(plan) == 40
        by_ticker: dict[str, set[str]] = {}
        for p in plan:
            by_ticker.setdefault(p.ticker, set()).add(p.slug)
        assert len(by_ticker) == 8
        assert all(v == TOPICS for v in by_ticker.values())

    def test_eight_companies_per_topic(self, fake_db):
        plan = ns.narrative_plan(max_per_company=5)
        counts: dict[str, int] = {}
        for p in plan:
            counts[p.slug] = counts.get(p.slug, 0) + 1
        assert counts == {t: 8 for t in TOPICS}

    def test_case_ids_are_unique(self, fake_db):
        plan = ns.narrative_plan(max_per_company=5)
        assert len({p.case_id for p in plan}) == 40

    def test_exactly_sixteen_references_are_missing_and_they_are_these(self, fake_db):
        missing = ns.missing_references(ns.narrative_plan(max_per_company=5))
        assert len(missing) == 16
        assert {m.case_id for m in missing} == EXPECTED_MISSING

    def test_expected_item_follows_the_seed_not_the_company(self, fake_db):
        by_slug = {s["slug"]: s["item"] for s in NARRATIVE_SEEDS}
        for p in ns.narrative_plan(max_per_company=5):
            assert p.item == by_slug[p.slug]


class TestGeneratorStillRefusesUnreferencedCases:
    def test_forty_cases_cannot_be_generated_yet(self, fake_db):
        from edgar_intel.evals.goldenset import generate_narrative_cases

        with pytest.raises(ValueError, match="Missing narrative reference"):
            generate_narrative_cases(max_per_company=5)

    def test_twenty_four_still_generate(self, fake_db):
        from edgar_intel.evals.goldenset import generate_narrative_cases

        cases = generate_narrative_cases(max_per_company=3)
        assert len(cases) == 24
        for case in cases:
            assert case.case_id in NARRATIVE_REFERENCES
            assert "REPLACE this reference" not in case.expected
            assert len(case.expected) > 80


class TestQuestionOverridesAreQuestions:
    def test_the_shipped_overrides_are_clean(self):
        assert check_question_overrides() == []

    def test_every_override_targets_a_real_case(self, fake_db):
        planned = {p.case_id for p in ns.narrative_plan(max_per_company=5)}
        assert set(NARRATIVE_QUESTION_OVERRIDES) <= planned

    def test_an_answer_shaped_override_is_caught(self):
        """The exact defect: reference prose in the question dict."""
        bad = {
            "nar-UNH-supply-concentration": (
                "UnitedHealth says noncompliance with privacy and security "
                "requirements, or a privacy or security breach involving the "
                "company or one of its third-party service providers, could harm "
                "its reputation and business. Consequences can include mandatory "
                "disclosure, loss of existing or new customers, increased "
                "incident-management and remediation costs, and significant "
                "fines, penalties and litigation awards."
            )
        }
        problems = check_question_overrides(bad, {})
        assert any("not a question" in p for p in problems)
        assert any("too long" in p for p in problems)

    def test_an_override_that_copies_a_reference_is_caught(self):
        problems = check_question_overrides({"nar-X": "same text?"}, {"nar-X": "same text?"})
        assert any("duplicates a reference answer" in p for p in problems)

    def test_the_override_is_used_as_the_question(self, fake_db):
        plan = {p.case_id: p for p in ns.narrative_plan(max_per_company=5)}
        unh = plan["nar-UNH-supply-concentration"]
        assert unh.question == NARRATIVE_QUESTION_OVERRIDES["nar-UNH-supply-concentration"]
        assert unh.question.endswith("?")

    def test_a_build_with_a_malformed_override_is_refused(self, monkeypatch):
        """The guard sits in `build`, so no path rebuilds the set past it."""
        import edgar_intel.evals.goldenset as gs

        monkeypatch.setattr(
            gs, "NARRATIVE_QUESTION_OVERRIDES", {"nar-X": "This is a statement."}
        )
        with pytest.raises(ValueError, match="malformed question overrides"):
            gs.build(path="/tmp/should-not-be-written.json")


class TestMissingSectionsAreNamedNotSkipped:
    def _item(self, case_id="nar-NVDA-legal", item="3"):
        return ns.NarrativePlanItem(
            case_id=case_id,
            cik="0001045810",
            ticker="NVDA",
            company="NVIDIA CORP",
            slug="legal",
            item=item,
            question="What material legal proceedings does NVIDIA CORP disclose?",
            fiscal_year=2026,
            has_reference=False,
        )

    def test_absent_expected_item_is_reported_with_what_is_present(self):
        sources = ns.CaseSources(
            plan=self._item(),
            accession="0001045810-26-000001",
            period_end=None,
            sections=[],
            available_items=[("1A", 120_000), ("7", 80_000)],
        )
        text = ns.render(sources)
        assert "NO SECTION FOR ITEM 3" in text
        assert "incorporated by reference" in text
        assert "1A (120,000c)" in text

    def test_a_filing_with_no_parsed_sections_says_so(self):
        sources = ns.CaseSources(
            plan=self._item(), accession="x", period_end=None, available_items=[]
        )
        assert "filing has no parsed sections" in ns.render(sources)

    def test_a_present_section_is_rendered_with_its_size(self):
        sources = ns.CaseSources(
            plan=self._item(),
            accession="x",
            period_end=None,
            sections=[ns.SectionText("3", "Legal Proceedings", 4, 1234, "body text")],
        )
        text = ns.render(sources)
        assert "--- SECTION 1 ---" in text
        assert "TITLE: Legal Proceedings" in text
        assert "CHARS: 1,234" in text
        assert "body text" in text
        assert "NO SECTION" not in text

    def test_the_header_says_whether_a_reference_exists(self):
        sources = ns.CaseSources(plan=self._item(), accession="x", period_end=None)
        assert "HAS REFERENCE: NO -- to be written" in ns.render(sources)

    def test_truncation_is_marked_never_silent(self):
        sources = ns.CaseSources(
            plan=self._item(),
            accession="x",
            period_end=None,
            sections=[ns.SectionText("3", "Legal Proceedings", 4, 40, "x" * 40)],
        )
        text = ns.render(sources, max_section_chars=10)
        assert "[TRUNCATED: showing 10 of 40 chars]" in text

    def test_no_truncation_by_default(self):
        sources = ns.CaseSources(
            plan=self._item(),
            accession="x",
            period_end=None,
            sections=[ns.SectionText("3", "t", 4, 40, "x" * 40)],
        )
        assert "TRUNCATED" not in ns.render(sources)


class TestTheYearOnTheTextIsTheTextsOwnYear:
    """The fiscal-year bug, in a new place: a year from one record, content from another."""

    def _sources(self, case_fy, filing_fy):
        item = ns.NarrativePlanItem(
            case_id="nar-COST-competition",
            cik="0000909832",
            ticker="COST",
            company="COSTCO WHOLESALE CORP /NEW",
            slug="competition",
            item="1A",
            question="q?",
            fiscal_year=case_fy,
            has_reference=False,
        )
        return ns.CaseSources(
            plan=item,
            accession="x",
            period_end=None,
            sections=[ns.SectionText("1A", "Risk Factors", 2, 9, "risk text")],
            filing_fiscal_year=filing_fy,
        )

    def test_a_newer_10q_year_against_older_10k_text_is_flagged(self):
        sources = self._sources(case_fy=2026, filing_fy=2025)
        assert sources.year_mismatch
        text = ns.render(sources)
        assert "WARNING: case year is FY2026" in text
        assert "FY2025 10-K" in text

    def test_agreement_is_silent(self):
        sources = self._sources(case_fy=2025, filing_fy=2025)
        assert not sources.year_mismatch
        assert "WARNING" not in ns.render(sources)

    def test_an_unknown_year_is_not_a_mismatch(self):
        assert not self._sources(case_fy=2025, filing_fy=None).year_mismatch
        assert not self._sources(case_fy=None, filing_fy=2025).year_mismatch

    def test_the_dump_summary_separates_mismatches_from_missing_sections(
        self, tmp_path, monkeypatch, fake_db
    ):
        def collect(item):
            return ns.CaseSources(
                plan=item,
                accession="x",
                period_end=None,
                sections=[ns.SectionText(item.item, "t", 1, 4, "text")],
                filing_fiscal_year=2024,
            )

        monkeypatch.setattr(ns, "collect_sources", collect)
        plan = [p for p in ns.narrative_plan(max_per_company=5) if p.ticker == "PG"]
        payload = ns.dump(plan, out_path=str(tmp_path / "d.txt"))
        assert payload["missing_sections"] == []
        assert len(payload["year_mismatches"]) == 5


class TestTheDumpCannotDestroyItsPredecessor:
    def test_the_name_encodes_the_selection(self):
        name = ns.default_dump_name(16, "missing")
        assert "16cases" in name and "missing" in name and name.endswith(".txt")

    def test_two_selections_do_not_collide(self):
        assert ns.default_dump_name(16, "missing") != ns.default_dump_name(40, "all")

    def test_an_existing_dump_is_not_overwritten(self, tmp_path, monkeypatch, fake_db):
        target = tmp_path / "sources.txt"
        target.write_text("the dump the 24 references were written from")
        plan = ns.narrative_plan(max_per_company=5)[:1]
        with pytest.raises(FileExistsError, match="provenance"):
            ns.dump(plan, out_path=str(target))
        assert "the 24 references" in target.read_text()

    def test_overwrite_is_explicit(self, tmp_path, monkeypatch, fake_db):
        monkeypatch.setattr(ns, "collect_sources", lambda item: ns.CaseSources(
            plan=item, accession="x", period_end=None,
            sections=[ns.SectionText(item.item, "t", 1, 9, "some text")],
        ))
        target = tmp_path / "sources.txt"
        target.write_text("old")
        payload = ns.dump(
            ns.narrative_plan(max_per_company=5)[:2], out_path=str(target), overwrite=True
        )
        assert payload["cases"] == 2
        assert "CASE ID:" in target.read_text()

    def test_the_summary_names_cases_whose_item_is_absent(self, tmp_path, monkeypatch, fake_db):
        def collect(item):
            found = item.slug != "legal"
            return ns.CaseSources(
                plan=item,
                accession="x",
                period_end=None,
                sections=(
                    [ns.SectionText(item.item, "t", 1, 5, "text")] if found else []
                ),
                available_items=[("1A", 10)],
            )

        monkeypatch.setattr(ns, "collect_sources", collect)
        plan = [p for p in ns.narrative_plan(max_per_company=5) if p.ticker == "NVDA"]
        payload = ns.dump(plan, out_path=str(tmp_path / "d.txt"))
        assert payload["missing_sections"] == ["nar-NVDA-legal"]
        assert payload["cases"] == 5
