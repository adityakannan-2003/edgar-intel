"""Case dissection: fact probes, gold-link grading, and stage attribution.

Grounded in the `nar-AAPL-legal` 24k arm, where 4 of 5 gold chunks reached the
prompt, the answer named every proceeding in the reference, and the judge failed
it three times out of three for stating "a violation of a 2021 Injunction ...
which is not mentioned in the reference" -- a phrase the reference contains.
"""

from __future__ import annotations

import pytest

from edgar_intel.evals.autopsy import (
    Autopsy,
    classify_gold_links,
    derive_fact_probes,
    diagnose,
    locate_fact_sources,
    parse_fact_spec,
    probe_text,
)

REFERENCE = (
    "Apple discloses EU Digital Markets Act proceedings, including a €500 million "
    "fine in the Article 5(4) investigation that Apple has appealed and a separate "
    "Article 6(4) investigation. It also discloses the DOJ and state attorneys "
    "general antitrust lawsuit alleging monopolization of smartphone markets, as "
    "well as the Epic Games litigation concerning App Store rules and the 2021 "
    "injunction."
)

ANSWER = (
    "Apple Inc. discloses several material legal proceedings, including: 1) A "
    "violation of a 2021 Injunction by the California District Court. 2) The "
    "Company is subject to investigations under the Digital Markets Act (DMA) by "
    "the European Commission, which has fined the Company €500 million. 3) A civil "
    "antitrust lawsuit filed by the Department of Justice and various state "
    "attorneys general alleging monopolization in the smartphone market. 4) A "
    "lawsuit filed by Epic Games alleging violations of antitrust laws related to "
    "the App Store."
)


class TestFactProbeDerivation:
    """Probes come out of the reference, not a second hand-written list."""

    def test_finds_the_currency_amount(self):
        probes = derive_fact_probes(REFERENCE)
        assert any("500" in label for label in probes)

    def test_finds_named_entities(self):
        probes = derive_fact_probes(REFERENCE)
        joined = " | ".join(probes)
        assert "Digital Markets Act" in joined
        assert "Epic Games" in joined

    def test_finds_statutory_references(self):
        probes = derive_fact_probes(REFERENCE)
        assert "Article 5(4)" in probes

    def test_finds_acronyms(self):
        assert "DOJ" in derive_fact_probes(REFERENCE)

    def test_explicit_spec_overrides(self):
        probes = parse_fact_spec("DMA=digital markets act|DMA; fine=€ ?500")
        assert probes == {"DMA": "digital markets act|DMA", "fine": "€ ?500"}

    def test_a_label_with_no_regex_matches_itself(self):
        assert parse_fact_spec("Epic Games") == {"Epic Games": r"Epic\ Games"}


class TestLiteralProbing:
    """Searched in the text. Never inferred from a gold label."""

    def test_present_facts_are_found_with_an_excerpt(self):
        found = probe_text(ANSWER, {"Epic Games": r"Epic\s+Games"})
        assert found["Epic Games"]["present"]
        assert "Epic Games" in found["Epic Games"]["excerpt"]

    def test_absent_facts_are_reported_absent(self):
        found = probe_text(ANSWER, {"Article 6(4)": r"Article\s+6\(4\)"})
        assert not found["Article 6(4)"]["present"]
        assert found["Article 6(4)"]["excerpt"] == ""

    def test_case_insensitive(self):
        assert probe_text("the digital markets act", {"x": "Digital Markets Act"})["x"]["present"]

    def test_the_answer_contains_every_proceeding_in_the_reference(self):
        """The measurement that moved the diagnosis off retrieval."""
        probes = {
            "DMA": r"Digital\s+Markets\s+Act",
            "fine": r"€\s?500",
            "DOJ": r"Department\s+of\s+Justice|DOJ",
            "Epic": r"Epic\s+Games",
            "2021 injunction": r"2021\s+Injunction",
        }
        found = probe_text(ANSWER, probes)
        assert all(v["present"] for v in found.values()), {
            k: v["present"] for k, v in found.items()
        }

    def test_the_reference_does_contain_the_2021_injunction(self):
        """The judge said it did not. It does, verbatim."""
        assert probe_text(REFERENCE, {"2021 injunction": r"2021\s+injunction"})[
            "2021 injunction"
        ]["present"]


class TestGoldLinkQuality:
    """"5/5 relevant" is not the same as "5 chunks that carry the answer"."""

    PROBES = {
        "DMA": r"Digital\s+Markets\s+Act",
        "fine": r"€\s?500",
        "DOJ": r"Department\s+of\s+Justice|DOJ",
        "Epic": r"Epic\s+Games",
    }

    def test_primary_evidence_carries_two_or_more_facts(self):
        bodies = {"a": "The European Commission fined the Company €500 million under the Digital Markets Act."}
        graded = classify_gold_links(["a"], bodies, self.PROBES)
        assert graded[0]["classification"] == "primary"
        assert graded[0]["n_facts"] == 2

    def test_supporting_evidence_carries_exactly_one(self):
        bodies = {"b": "Epic Games filed suit regarding App Store rules."}
        assert classify_gold_links(["b"], bodies, self.PROBES)[0]["classification"] == "supporting"

    def test_weak_evidence_carries_none(self):
        """A Risk Factors chunk saying litigation is a risk is not evidence."""
        bodies = {"c": "The Company is subject to various legal proceedings which could be material."}
        graded = classify_gold_links(["c"], bodies, self.PROBES)
        assert graded[0]["classification"] == "weak"
        assert graded[0]["facts_matched"] == []

    def test_an_unretrieved_gold_chunk_is_marked(self):
        graded = classify_gold_links(["missing"], {}, self.PROBES)
        assert graded[0]["retrieved"] is False
        assert graded[0]["classification"] == "weak"

    def test_fact_sources_name_the_chunk(self):
        bodies = {"a": "Epic Games sued.", "b": "unrelated text"}
        sources = locate_fact_sources({"Epic": r"Epic\s+Games"}, bodies)
        assert sources["Epic"] == ["a"]


class TestStageAttribution:
    def _report(self, **kw) -> Autopsy:
        base = dict(
            case_id="nar-AAPL-legal",
            git_sha="8a10f1f",
            config={"context_max_chars": 24000},
            question="What material legal proceedings does Apple Inc. disclose?",
            reference=REFERENCE,
            answer=ANSWER,
            gold_ids=["57410", "57411", "57396", "57356", "57355"],
            max_completion_tokens=600,
            completion_tokens=222,
        )
        base.update(kw)
        return Autopsy(**base)

    def test_missing_from_prompt_blames_retrieval(self):
        report = self._report(
            gold_included=["57410"],
            facts_in_prompt={"Epic": {"present": False, "pattern": "", "excerpt": ""}},
            facts_in_answer={},
        )
        assert "RETRIEVAL/PACKING" in diagnose(report)

    def test_in_prompt_but_not_in_answer_blames_generation(self):
        report = self._report(
            gold_included=["57410"],
            facts_in_prompt={"Epic": {"present": True, "pattern": "", "excerpt": "x"}},
            facts_in_answer={"Epic": {"present": False, "pattern": "", "excerpt": ""}},
        )
        text = diagnose(report)
        assert "GENERATION" in text
        assert "ANSWER_SYSTEM" in text

    def test_facts_everywhere_and_still_failing_blames_the_judge(self):
        """The `nar-AAPL-legal` situation exactly."""
        present = {"present": True, "pattern": "", "excerpt": "x"}
        report = self._report(
            passed=False,
            gold_included=["57410", "57411", "57396", "57356"],
            gold_excluded_by_cap=["57355"],
            facts_in_prompt={"DMA": present, "DOJ": present, "Epic": present},
            facts_in_answer={"DMA": present, "DOJ": present, "Epic": present},
        )
        text = diagnose(report)
        assert "JUDGE" in text
        assert "Retrieval work cannot move this number" in text

    def test_a_pass_is_reported_as_a_pass(self):
        present = {"present": True, "pattern": "", "excerpt": "x"}
        report = self._report(
            passed=True,
            facts_in_prompt={"DMA": present},
            facts_in_answer={"DMA": present},
        )
        assert "PASS" in diagnose(report)

    def test_hitting_the_token_ceiling_is_flagged(self):
        present = {"present": True, "pattern": "", "excerpt": "x"}
        report = self._report(
            completion_tokens=600,
            facts_in_prompt={"DMA": present},
            facts_in_answer={"DMA": present},
            passed=True,
        )
        assert "token ceiling" in diagnose(report)

    def test_a_unanimous_judge_failure_is_called_out(self):
        present = {"present": True, "pattern": "", "excerpt": "x"}
        report = self._report(
            passed=False,
            facts_in_prompt={"DMA": present},
            facts_in_answer={"DMA": present},
        )
        runs = [{"verdict": False} for _ in range(3)]
        text = diagnose(report, runs)
        assert "3/3" in text
        assert "measuring the reference, not the answer" in text

    def test_weak_gold_links_are_reported(self):
        present = {"present": True, "pattern": "", "excerpt": "x"}
        report = self._report(
            passed=True,
            facts_in_prompt={"DMA": present},
            facts_in_answer={"DMA": present},
            gold_link_quality=[
                {"chunk_id": "57355", "classification": "weak", "facts_matched": [], "n_facts": 0},
                {"chunk_id": "57410", "classification": "primary", "facts_matched": ["a", "b"], "n_facts": 2},
            ],
        )
        text = diagnose(report)
        assert "57355" in text
        assert "inflates the retrieval" in text


class TestReportNaming:
    """The 20k run was overwritten by the 24k run. Evidence, destroyed silently."""

    def test_name_encodes_the_arm(self):
        from edgar_intel.evals.context_probe import default_report_name

        assert default_report_name(24000, 2.0, (False,), "greedy-stop").endswith(
            "context_probe_24k_boost2_norerank.json"
        )

    def test_different_context_sizes_get_different_names(self):
        from edgar_intel.evals.context_probe import default_report_name

        names = {
            default_report_name(n, 2.0, (False,), "greedy-stop")
            for n in (12000, 16000, 20000, 24000)
        }
        assert len(names) == 4

    def test_boost_and_packing_are_distinguished(self):
        from edgar_intel.evals.context_probe import default_report_name

        assert default_report_name(12000, 0.0, (False,), "greedy-stop") != default_report_name(
            12000, 2.0, (False,), "greedy-stop"
        )
        assert default_report_name(12000, 2.0, (False,), "skip-oversized") != default_report_name(
            12000, 2.0, (False,), "greedy-stop"
        )

    def test_probe_refuses_to_overwrite(self):
        import os
        import tempfile

        from edgar_intel.evals.context_probe import probe

        with tempfile.TemporaryDirectory() as tmp:
            existing = os.path.join(tmp, "taken.json")
            with open(existing, "w", encoding="utf-8") as fh:
                fh.write("{}")
            with pytest.raises(FileExistsError):
                probe([], out_path=existing)

    def test_overwrite_flag_allows_it(self):
        import os
        import tempfile

        from edgar_intel.evals.context_probe import probe

        with tempfile.TemporaryDirectory() as tmp:
            existing = os.path.join(tmp, "taken.json")
            with open(existing, "w", encoding="utf-8") as fh:
                fh.write("{}")
            payload = probe([], out_path=existing, overwrite=True)
            assert payload["out_path"] == existing


class TestTraceCapturesLiteralBytes:
    def test_answer_question_accepts_a_trace(self):
        import inspect

        from edgar_intel.evals.runner import answer_question

        assert "trace" in inspect.signature(answer_question).parameters

    def test_the_prompt_is_rendered_once(self):
        """A reconstructed prompt is a second implementation of the formatting."""
        import ast
        import inspect

        from edgar_intel.evals import runner

        tree = ast.parse(inspect.getsource(runner))
        renders = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ANSWER_TEMPLATE"
        ]
        assert len(renders) == 1
