"""`index build` will not silently replace the served index.

Without --suffix, a build deletes and re-creates a strategy's index in place.
The served strategy's index is what the golden set's labels and the reference
run point at, and it predates D9, so today's chunker does not reproduce it.
On 6 Oct the rebuild scored 0.784 numeric accuracy against the shipped index's
0.837. One `edgar-intel index build --strategy all` on the local database would
have thrown that away. A fresh database still builds as before: CI's eval-gate
job and a new deployment both run `index build` with nothing to lose.

No database: the chunk count and the build itself are stubbed.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from edgar_intel.retrieval import index as index_mod


@pytest.fixture
def served_chunks(monkeypatch):
    """Set how many chunks the served strategy already has."""

    def install(n: int) -> None:
        monkeypatch.setattr(index_mod.db, "query_one", lambda sql, params=None: {"n": n})

    return install


class TestTheProblem:
    def test_an_existing_served_index_is_not_replaced_by_default(self, served_chunks):
        served_chunks(5468)
        problem = index_mod.replacement_problem(["section_aware"], "", False)
        assert problem and "served index (5468 chunks)" in problem
        assert "--suffix" in problem and "--replace-served" in problem

    def test_all_includes_the_served_strategy(self, served_chunks):
        served_chunks(5468)
        assert index_mod.replacement_problem(
            ["fixed", "recursive", "section_aware", "semantic"], "", False
        )

    def test_a_suffix_builds_beside_it(self, served_chunks):
        served_chunks(5468)
        assert index_mod.replacement_problem(["section_aware"], "_t600", False) is None

    def test_the_flag_allows_it(self, served_chunks):
        served_chunks(5468)
        assert index_mod.replacement_problem(["section_aware"], "", True) is None

    def test_a_fresh_database_builds_as_before(self, served_chunks):
        """CI's eval-gate job: `index build --strategy section_aware`, empty DB."""
        served_chunks(0)
        assert index_mod.replacement_problem(["section_aware"], "", False) is None

    def test_other_strategies_are_not_guarded(self, served_chunks):
        served_chunks(5468)
        assert index_mod.replacement_problem(["fixed"], "", False) is None


class TestTheCommand:
    @pytest.fixture
    def built(self, monkeypatch) -> list[str]:
        seen: list[str] = []

        def fake_build(strategy, *a, label=None, **k):
            seen.append(label or strategy)
            return index_mod.IndexStats(strategy=label or strategy)

        monkeypatch.setattr(index_mod, "index_strategy", fake_build)
        return seen

    def run(self, *args):
        from edgar_intel.cli import app

        return CliRunner().invoke(app, ["index", "build", *args])

    def test_refuses_and_builds_nothing(self, served_chunks, built):
        served_chunks(5468)
        result = self.run("--strategy", "section_aware")
        assert result.exit_code == 1
        assert built == []

    def test_builds_with_the_flag(self, served_chunks, built):
        served_chunks(5468)
        result = self.run("--strategy", "section_aware", "--replace-served")
        assert result.exit_code == 0, result.output
        assert built == ["section_aware"]

    def test_builds_beside_it_with_a_suffix(self, served_chunks, built):
        served_chunks(5468)
        result = self.run("--strategy", "section_aware", "--suffix", "_t600")
        assert result.exit_code == 0, result.output
        assert built == ["section_aware_t600"]
