"""The chunk size was picked for a 512-token budget. The embedder's window is 256.

`index report` averages `token_est`, which is `len(body) // 4` -- a character
heuristic. `section_aware` averages 473 of those, so ~1,892 characters per chunk,
while all-MiniLM-L6-v2 reads 256 word-piece tokens and `LocalEmbedder.embed`
never sets `max_seq_length`. Text past the window is dropped before the vector
exists: present in Postgres, reachable lexically, invisible to dense retrieval.

The same reasoning was already applied to the reranker -- `rerank_max_length` set
explicitly, a budget for `[CLS]` and two `[SEP]`, a comment about scoring a
passage the model has only partly read. The embedder got none of it.

These tests cover the *reporting* of that measurement, not the measurement
itself: the tokeniser and the model cannot load in CI. What they pin is that the
verdict says "no problem" when there is none, and never draws a conclusion when
the window could not be read -- because a confident verdict from a failed
measurement is how this project's worst defects survived.
"""

from __future__ import annotations

from edgar_intel.retrieval.index import embed_window_verdict


def _audit(window: int, rows: list[dict]) -> dict:
    return {"model": "m", "window_tokens": window, "rows": rows}


def _row(strategy: str, unseen: float, over_pct: float = 50.0, median: int = 400) -> dict:
    return {
        "strategy": strategy,
        "sampled": 300,
        "median_tokens": median,
        "p95_tokens": median + 100,
        "max_tokens": median + 200,
        "over_window": int(3 * over_pct),
        "over_window_pct": over_pct,
        "text_unseen_pct": unseen,
    }


class TestTheVerdictWhenTruncationIsReal:
    def test_it_names_the_share_of_text_never_embedded(self):
        text = embed_window_verdict(_audit(256, [_row("section_aware", 44.2)]))
        assert "44.2%" in text
        assert "256-token window" in text
        assert "never reaches the embedder" in text

    def test_it_names_the_worst_strategy_not_the_first(self):
        audit = _audit(
            256,
            [_row("fixed", 12.0), _row("semantic", 71.5), _row("section_aware", 44.2)],
        )
        text = embed_window_verdict(audit)
        assert "semantic" in text
        assert "71.5%" in text

    def test_it_blocks_the_strategy_comparison(self):
        """Four strategies handicapped the same way do not rank meaningfully."""
        text = embed_window_verdict(_audit(256, [_row("section_aware", 44.2)]))
        assert "before running the strategy comparison" in text
        assert "handicapped" in text

    def test_it_prescribes_re_chunking_and_re_measuring(self):
        text = embed_window_verdict(_audit(256, [_row("section_aware", 44.2)]))
        assert "Re-chunk" in text
        assert "re-measure hit@5" in text


class TestTheVerdictWhenThereIsNoProblem:
    def test_chunks_inside_the_window_clear_d8(self):
        """The measurement has to be able to come back negative, or it is theatre."""
        text = embed_window_verdict(_audit(256, [_row("section_aware", 0.0, 0.0, 180)]))
        assert "not a defect" in text
        assert "comparison can run" in text

    def test_a_rounding_sliver_still_counts_as_fitting(self):
        text = embed_window_verdict(_audit(256, [_row("fixed", 0.4, 1.0, 200)]))
        assert "not a defect" in text


class TestAFailedMeasurementDrawsNoConclusion:
    def test_an_unreadable_window_refuses_to_conclude(self):
        """A confident verdict from a broken instrument is this project's worst bug."""
        text = embed_window_verdict(_audit(0, [_row("section_aware", 0.0)]))
        assert "do not conclude anything" in text
        assert "not a defect" not in text

    def test_no_sampled_chunks_says_so(self):
        assert "No chunks sampled." in embed_window_verdict(_audit(256, []))


class TestItIsWiredAsACommand:
    """The last measurement kept as a one-off script was lost with the script."""

    @staticmethod
    def _cli() -> str:
        import pathlib

        return pathlib.Path("src/edgar_intel/cli.py").read_text()

    def test_the_command_exists(self):
        assert '@index_app.command("embed-window")' in self._cli()

    def test_it_reports_the_window_and_the_verdict(self):
        cli = self._cli()
        assert "embed_window_audit" in cli
        assert "embed_window_verdict" in cli

    def test_the_sample_is_seeded_so_the_number_is_reproducible(self):
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path("src/edgar_intel/cli.py").read_text())
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "index_embed_window"
        )
        assert {a.arg for a in fn.args.args} >= {"strategy", "sample", "seed"}


# ------------------------------------------------------------ the measurement
class _FakeTokenizer:
    """One word-piece per whitespace token, plus [CLS] and [SEP]."""

    def __init__(self) -> None:
        self.kwargs: list[dict] = []

    def encode(self, text: str, **kwargs) -> list[int]:
        self.kwargs.append(kwargs)
        return [0] * (len(text.split()) + (2 if kwargs.get("add_special_tokens") else 0))


class _FakeEncoder:
    max_seq_length = 256

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()


def _patch(monkeypatch, rows: list[dict]) -> tuple[list, _FakeEncoder]:
    import edgar_intel.db as db
    import edgar_intel.providers.local as local

    calls: list = []
    enc = _FakeEncoder()

    def fake_query(sql, params=None):
        calls.append((sql, params))
        return rows

    monkeypatch.setattr(db, "query", fake_query)
    monkeypatch.setattr(local, "_load_encoder", lambda name: enc)
    return calls, enc


class TestTheAuditItself:
    """The tokeniser cannot load here, but the arithmetic around it can be pinned."""

    def test_the_sample_is_per_strategy_not_one_limit_across_all_four(self, monkeypatch):
        """One LIMIT across four corpora split 300 rows by corpus size, while the
        help text promised 300 per corpus."""
        from edgar_intel.retrieval.index import embed_window_audit

        calls, _ = _patch(monkeypatch, [])
        embed_window_audit(None, sample=300, seed=7)
        sql, params = calls[0]
        assert "PARTITION BY strategy" in sql
        assert "rn <= %s" in sql
        assert params == ["7", 300]

    def test_one_strategy_is_filtered_inside_the_window(self, monkeypatch):
        from edgar_intel.retrieval.index import embed_window_audit

        calls, _ = _patch(monkeypatch, [])
        embed_window_audit("section_aware", sample=50, seed=3)
        sql, params = calls[0]
        assert "WHERE strategy = %s" in sql
        assert params == ["3", "section_aware", 50]

    def test_the_tokeniser_warning_about_its_own_512_is_silenced(self, monkeypatch):
        """HF warns "(n > 512)" -- the tokeniser's limit, not the model's 256."""
        from edgar_intel.retrieval.index import embed_window_audit

        _, enc = _patch(monkeypatch, [{"strategy": "fixed", "body": "a b c", "token_est": 1}])
        embed_window_audit()
        assert enc.tokenizer.kwargs == [{"add_special_tokens": True, "verbose": False}]

    def test_unseen_share_and_over_window_counts(self, monkeypatch):
        from edgar_intel.retrieval.index import embed_window_audit

        rows = [
            {"strategy": "section_aware", "body": "w " * 298, "token_est": 250},  # 300 wp
            {"strategy": "section_aware", "body": "w " * 198, "token_est": 200},  # 200 wp
            {"strategy": "section_aware", "body": "w " * 510, "token_est": 400},  # 512 wp
        ]
        _patch(monkeypatch, rows)
        audit = embed_window_audit()
        row = audit["rows"][0]
        assert audit["window_tokens"] == 256
        assert row["over_window"] == 2
        assert row["over_window_pct"] == 66.7
        # seen = 256 + 200 + 256 = 712 of 1,012
        assert row["text_unseen_pct"] == 29.6


class TestTheSuggestedTarget:
    """The 4-chars-per-token rule undercounts figures, so "about 250 token_est"
    is a guess the measurement should replace with a number."""

    def test_it_is_the_window_over_the_p95_density(self):
        from edgar_intel.retrieval.index import window_summary

        # word-pieces per token_est (excluding [CLS]/[SEP]): 1.0, 1.25, 1.5, 2.0
        pairs = [(202, 200), (252, 200), (302, 200), (402, 200)]
        out = window_summary({"fixed": pairs}, 256)
        row = out["rows"][0]
        assert row["wp_per_token_est_p95"] == 1.5
        assert row["suggested_target_tokens"] == int(254 / 1.5)  # 169

    def test_tiny_chunks_do_not_set_the_ratio(self):
        """"Item 1." is 1 token_est and several word-pieces."""
        from edgar_intel.retrieval.index import window_summary

        out = window_summary({"semantic": [(6, 1), (5, 1), (202, 200)]}, 256)
        assert out["rows"][0]["wp_per_token_est_p95"] == 1.0

    def test_one_ceiling_for_every_strategy_is_the_most_conservative(self):
        from edgar_intel.retrieval.index import window_summary

        out = window_summary(
            {"fixed": [(202, 200)], "semantic": [(402, 200)]}, 256
        )
        assert out["suggested_target_tokens"] == min(
            r["suggested_target_tokens"] for r in out["rows"]
        )

    def test_the_verdict_names_it(self):
        audit = _audit(256, [_row("section_aware", 44.2)])
        audit["suggested_target_tokens"] = 169
        text = embed_window_verdict(audit)
        assert "about 169 token_est" in text
        assert "--target-tokens" in text

    def test_no_window_means_no_suggestion(self):
        from edgar_intel.retrieval.index import window_summary

        assert window_summary({"fixed": [(300, 250)]}, 0)["suggested_target_tokens"] is None
