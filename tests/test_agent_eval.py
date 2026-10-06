"""The agent on the golden set: graded by the same code as retrieve-then-answer.

`eval agent` exists to compare the pipeline `/ask` serves with the one every
accuracy figure measured. That comparison is only single-variable if the
grading is identical, so these tests pin it to `build_result`, and pin the one
rule the agent adds: only what `/ask` returns as an answer is graded as one.
An escalation that holds back a correct draft is still not a pass.

No database, no network: the model is scripted, the tools are stubbed, and the
database calls the harness makes are captured.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from edgar_intel import db
from edgar_intel.agent import loop as agent_loop
from edgar_intel.agent.tools import ToolResult
from edgar_intel.evals import agent_eval, judge, report
from edgar_intel.evals.judge import JudgeVerdict, build_result, grade_numeric
from edgar_intel.evals.runner import RunAborted
from edgar_intel.evals.schemas import EvalCase
from edgar_intel.providers.base import Completion

ACCESSION = "0000080424-24-000083"
PG_NET_INCOME = EvalCase(
    case_id="num-PG-NetIncomeLoss-2024",
    kind="numeric",
    question="How much net income did PROCTER & GAMBLE Co report in FY2024?",
    expected="$14.88 billion",
    expected_value=14_879_000_000.0,
    unit="USD",
    ticker="PG",
    fiscal_year=2024,
    tag="NetIncomeLoss",
)
PG_YOY = EvalCase(
    case_id="yoy-PG-NetIncomeLoss-2023-2024",
    kind="numeric",
    question="How did PROCTER & GAMBLE Co's net income change from FY2023 to FY2024?",
    expected="increased 0.4%",
    expected_value=0.4,
    unit="percent",
    ticker="PG",
    difficulty="comparative",
)
PG_RISKS = EvalCase(
    case_id="nar-PG-supply-concentration",
    kind="narrative",
    question="What supply chain risks does PROCTER & GAMBLE Co disclose?",
    expected="P&G discloses reliance on third-party suppliers.",
    ticker="PG",
)

FACT = ToolResult(
    ok=True,
    summary=f"PROCTER & GAMBLE Co Net income FY2024: $14.88B (source: XBRL {ACCESSION})",
    citations=[f"xbrl:{ACCESSION}"],
)
# Consolidated net earnings, noncontrolling interests included: the line the
# retrieve-then-answer path picked in three cases.
NCI_TEXT = ToolResult(
    ok=True,
    summary="[c9] PG FY2024 Item 8\nNet earnings 14,974 ... attributable to P&G 14,879",
    citations=["c9"],
)
PASSAGES = ToolResult(
    ok=True,
    summary="[c1] PG FY2024 Item 1A\nWe rely on third-party suppliers for materials.",
    citations=["c1"],
)


def tool(name: str, **args) -> dict:
    return {"thought": f"call {name}", "tool": name, "args": args}


def answer(text: str, citations: list[str] | None = None, confidence: float = 0.9) -> dict:
    return {"thought": "done", "answer": text, "citations": citations or [],
            "confidence": confidence}


class ScriptedPerQuestion:
    """Replies scripted per question, read from the transcript's first line."""

    name = "scripted"

    def __init__(self, scripts: dict[str, list[dict | str]]) -> None:
        self.scripts = scripts
        self.calls: dict[str, int] = {}

    def complete(self, prompt, *, system=None, model=None, temperature=0.0,
                 max_tokens=1024, json_schema=None) -> Completion:
        question = prompt.split("\n", 1)[0].removeprefix("QUESTION: ")
        replies = self.scripts[question]
        i = self.calls.get(question, 0)
        self.calls[question] = i + 1
        reply = replies[min(i, len(replies) - 1)]
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return Completion(text=text, prompt_tokens=1_000, completion_tokens=100, model="scripted")


@pytest.fixture
def agent(monkeypatch):
    """Script the agent's model and tools. Persisting a trace is an error."""

    def _apply(scripts: dict[str, list], tool_results: dict[str, ToolResult] | None = None):
        llm = ScriptedPerQuestion(scripts)
        monkeypatch.setattr(agent_loop, "get_llm", lambda: llm)

        def no_persist(run):
            raise AssertionError("an evaluation run must not write agent_traces by default")

        monkeypatch.setattr(agent_loop, "_persist", no_persist)
        results = tool_results or {}
        monkeypatch.setattr(agent_loop, "call_tool", lambda name, args: results[name])
        monkeypatch.setattr(agent_loop, "TOOLS", {
            "search_filings": {}, "get_fact": {}, "compare_fact": {}, "escalate_to_human": {},
        })
        return llm

    return _apply


def run_one(case: EvalCase, script: list, tools: dict[str, ToolResult], agent):
    agent({case.question: script}, tools)
    run = agent_loop.run_agent(case.question, persist=False)
    return run, *agent_eval.grade_agent_run(case, run)


class TestAnsweredRunsUseTheRetrieveThenAnswerGrader:
    def test_the_verdict_is_build_results_verdict(self, agent):
        run, result, record = run_one(
            PG_NET_INCOME,
            [tool("get_fact", ticker="PG", tag="NetIncomeLoss", fiscal_year=2024),
             answer("Net income was $14.88 billion.", [f"xbrl:{ACCESSION}"])],
            {"get_fact": FACT}, agent,
        )
        reference = build_result(
            PG_NET_INCOME, run.answer, {}, run.total_latency_ms,
            run.prompt_tokens, run.completion_tokens,
        )
        assert run.outcome == "answered"
        assert result.passed is True
        assert (result.passed, result.score, result.judge_rationale, result.abstained) == (
            reference.passed, reference.score, reference.judge_rationale, reference.abstained
        )
        assert record["tools"] == ["get_fact"]
        assert record["steps"] == 2

    def test_a_wrong_line_is_a_wrong_answer_not_an_abstention(self, agent):
        _, result, _ = run_one(
            PG_NET_INCOME,
            [tool("search_filings", query="P&G net earnings 2024"),
             answer("Net income was $14,974 million.", ["c9"])],
            {"search_filings": NCI_TEXT}, agent,
        )
        assert result.passed is False
        assert result.abstained is False

    def test_saying_the_evidence_is_missing_is_an_abstention(self, agent):
        _, result, record = run_one(
            PG_NET_INCOME,
            [tool("search_filings", query="P&G net income"),
             answer("The tools do not provide P&G's FY2024 net income.")],
            {"search_filings": PASSAGES}, agent,
        )
        assert record["outcome"] == "answered"
        assert result.passed is False
        assert result.abstained is True

    def test_the_judge_reads_what_the_agent_read(self, agent, monkeypatch):
        seen = {}

        def fake_judge(case, answer, source_context="", **_):
            seen["context"] = source_context
            return JudgeVerdict(verdict=True, rationale="matches", prompt_tokens=500,
                                completion_tokens=50)

        monkeypatch.setattr(judge, "judge_narrative", fake_judge)
        run, result, record = run_one(
            PG_RISKS,
            [tool("search_filings", query="supplier risk", ticker="PG"),
             answer("P&G relies on third-party suppliers.", ["c1"])],
            {"search_filings": PASSAGES}, agent,
        )
        assert seen["context"] == PASSAGES.summary
        assert run.evidence == [PASSAGES.summary]
        assert result.passed is True
        # The judge's tokens are the eval's cost, not the agent's.
        assert record["judge_cost_usd"] > 0
        assert record["agent_cost_usd"] == pytest.approx(
            (2_000 * 0.15 + 200 * 0.60) / 1_000_000
        )


class TestOnlyAnAnswerIsGradedAsOne:
    def test_a_correct_draft_held_back_is_not_a_pass(self, agent):
        run, result, record = run_one(
            PG_NET_INCOME,
            [tool("get_fact", ticker="PG", tag="NetIncomeLoss", fiscal_year=2024),
             answer("Net income was $14.88 billion.", [f"xbrl:{ACCESSION}"], confidence=0.4)],
            {"get_fact": FACT}, agent,
        )
        assert run.outcome == "escalated"
        # The text /ask returns contains the right figure, and the numeric
        # grader would pass it. That is exactly why it is not handed over.
        assert grade_numeric(PG_NET_INCOME, run.answer)[0] is True
        assert result.passed is False
        assert result.abstained is True
        assert result.judge_rationale.startswith("NOT ANSWERED")
        assert record["draft_answer"] == "Net income was $14.88 billion."
        assert record["draft_passed"] is True

    def test_a_wrong_draft_is_recorded_as_wrong(self, agent):
        _, _, record = run_one(
            PG_NET_INCOME,
            [tool("search_filings", query="P&G net earnings"),
             answer("Net income was $14,974 million.", ["c9"], confidence=0.3)],
            {"search_filings": NCI_TEXT}, agent,
        )
        assert record["outcome"] == "escalated"
        assert record["draft_passed"] is False

    def test_an_exhausted_run_is_declined_with_no_draft(self, agent):
        _, result, record = run_one(
            PG_NET_INCOME,
            [tool("search_filings", query="net income"),
             tool("search_filings", query="net income 2024"),
             tool("search_filings", query="net earnings"),
             tool("search_filings", query="net earnings 2024"),
             tool("search_filings", query="income statement"),
             tool("search_filings", query="earnings per share"),
             tool("search_filings", query="consolidated earnings"),
             tool("search_filings", query="statement of earnings")],
            {"search_filings": PASSAGES}, agent,
        )
        assert record["outcome"] == "exhausted"
        assert record["steps"] == 8
        assert result.passed is False and result.abstained is True
        assert record["draft_passed"] is None

    def test_an_escalated_narrative_is_not_sent_to_the_judge(self, agent, monkeypatch):
        def no_judge(*a, **k):
            raise AssertionError("an escalation is not an answer to judge")

        monkeypatch.setattr(judge, "judge_narrative", no_judge)
        _, result, record = run_one(
            PG_RISKS,
            [tool("escalate_to_human", reason="no passages on suppliers")],
            {"escalate_to_human": ToolResult(ok=True, summary="ESCALATED: no passages")},
            agent,
        )
        assert record["outcome"] == "escalated"
        assert result.passed is False
        assert result.abstained is False  # abstention is a numeric-only notion
        assert record["judge_cost_usd"] == 0


class FakeDB:
    """The database calls `run_agent_suite` makes, captured."""

    def __init__(self, chunks: int = 10) -> None:
        self.chunks = chunks
        self.runs: list[tuple] = []
        self.executed: list[tuple[str, tuple]] = []

    def query_one(self, sql, params=None):
        if "FROM chunks" in sql:
            return {"chunks": self.chunks, "embedded": self.chunks, "max_token_est": 600,
                    "avg_token_est": 400, "min_id": 1, "max_id": self.chunks}
        if "INSERT INTO eval_runs" in sql:
            self.runs.append(params)
            return {"id": 7}
        raise AssertionError(f"unexpected query_one: {sql}")

    def query(self, sql, params=None):
        return []  # no human labels, so no calibration pairs

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return 1

    def agent_records(self) -> list[dict]:
        return [json.loads(p[0]) for s, p in self.executed if "SET agent" in s]

    def final_summary(self) -> dict:
        return json.loads([p for s, p in self.executed if "SET finished_at" in s][-1][0])


@pytest.fixture
def fake_db(monkeypatch, tmp_path):
    fake = FakeDB()
    for name in ("query_one", "query", "execute"):
        monkeypatch.setattr(db, name, getattr(fake, name))
    monkeypatch.setenv("EDGAR_REPORTS_DIR", str(tmp_path))
    return fake


class TestTheSuite:
    SCRIPTS = {
        PG_NET_INCOME.question: [
            tool("get_fact", ticker="PG", tag="NetIncomeLoss", fiscal_year=2024),
            answer("Net income was $14.88 billion.", [f"xbrl:{ACCESSION}"]),
        ],
        PG_YOY.question: [
            tool("compare_fact", ticker="PG", tag="NetIncomeLoss", year_a=2023, year_b=2024),
            answer("It may have increased.", confidence=0.3),
        ],
        PG_RISKS.question: [tool("escalate_to_human", reason="nothing found")],
    }
    TOOLS = {
        "get_fact": FACT,
        "compare_fact": ToolResult(ok=True, summary="PG NetIncomeLoss: FY2023 -> FY2024, increased 0.4%"),
        "escalate_to_human": ToolResult(ok=True, summary="ESCALATED: nothing found"),
    }

    def test_records_a_marked_run_with_every_case(self, agent, fake_db, tmp_path):
        agent(self.SCRIPTS, self.TOOLS)
        run_id, summary = agent_eval.run_agent_suite(
            [PG_NET_INCOME, PG_YOY, PG_RISKS], label="agent-test", sha="abc1234"
        )
        assert run_id == 7
        run_key, sha, label, config, n = fake_db.runs[0]
        config = json.loads(config)
        assert (sha, label, n) == ("abc1234", "agent-test", 3)
        assert config["pipeline"] == "agent"
        assert config["agent_max_steps"] == 8
        assert config["agent_confidence_floor"] == 0.55
        assert config["search"]["use_rerank"] is False
        assert config["index"]["chunks"] == 10

        records = fake_db.agent_records()
        assert [r["outcome"] for r in records] == ["answered", "escalated", "escalated"]

        agent_block = summary["agent"]
        assert agent_block["outcomes"] == {"answered": 1, "escalated": 2}
        assert agent_block["escalation_rate"] == pytest.approx(2 / 3, abs=1e-4)
        assert agent_block["by_kind"]["single_hop"] == {
            "n": 1, "passed": 1, "escalated": 0, "not_answered": 0, "accuracy": 1.0,
        }
        assert agent_block["by_kind"]["comparative"]["escalated"] == 1
        assert agent_block["low_confidence_drafts_graded"] == 1
        assert agent_block["low_confidence_drafts_correct"] == 0
        # Escalations are declines, never hallucinations.
        assert summary["numeric_accuracy"] == 0.5
        assert summary["abstention_rate"] == 0.5
        assert summary["hallucination_rate"] == 0.0
        assert fake_db.final_summary()["agent"]["outcomes"] == agent_block["outcomes"]

        report_file = json.loads((tmp_path / f"{summary['run_key']}.json").read_text())
        assert [c["case_id"] for c in report_file["cases"]] == [
            PG_NET_INCOME.case_id, PG_YOY.case_id, PG_RISKS.case_id,
        ]
        assert report_file["cases"][0]["trace"][0]["tool"] == "get_fact"

    def test_a_dead_endpoint_voids_the_run(self, fake_db, monkeypatch):
        def unauthorized(question, persist=True):
            raise RuntimeError("401 Unauthorized: invalid_api_key")

        monkeypatch.setattr(agent_eval, "run_agent", unauthorized)
        with pytest.raises(RunAborted, match="void"):
            agent_eval.run_agent_suite([PG_NET_INCOME] * 10, label="agent-test", sha="x")
        aborted = json.loads([p for s, p in fake_db.executed if "SET finished_at" in s][-1][0])
        assert aborted["aborted"] == "infrastructure"

    def test_an_empty_index_stops_before_any_model_call(self, fake_db, monkeypatch):
        fake_db.chunks = 0
        monkeypatch.setattr(agent_eval, "run_agent", lambda *a, **k: pytest.fail("model called"))
        with pytest.raises(RunAborted, match="empty or partial index"):
            agent_eval.run_agent_suite([PG_NET_INCOME], label="agent-test", sha="x")
        assert fake_db.runs == []


class TestLatestRunIsPerPipeline:
    """An agent run must not become what `/stats/eval` serves or the gate reads."""

    @staticmethod
    def _capture(monkeypatch) -> list:
        calls: list = []
        monkeypatch.setattr(db, "query_one", lambda sql, params=None: calls.append((sql, params)))
        return calls

    def test_the_default_is_retrieve_then_answer(self, monkeypatch):
        calls = self._capture(monkeypatch)
        report.latest_run()
        sql, params = calls[0]
        assert "COALESCE(config->>'pipeline', 'rag') = %s" in sql
        assert params == ["rag"]

    def test_a_label_still_filters_by_pipeline(self, monkeypatch):
        calls = self._capture(monkeypatch)
        report.latest_run("baseline")
        assert calls[0][1] == ["rag", "baseline"]

    def test_agent_runs_are_asked_for_by_name(self, monkeypatch):
        calls = self._capture(monkeypatch)
        report.latest_run(pipeline="agent")
        assert calls[0][1] == ["agent"]


class TestLoopExposesWhatTheModelRead:
    def test_evidence_is_every_tool_output_in_order(self, agent):
        agent(
            {PG_NET_INCOME.question: [
                tool("search_filings", query="net income"),
                tool("get_fact", ticker="PG", tag="NetIncomeLoss", fiscal_year=2024),
                answer("Net income was $14.88 billion.", [f"xbrl:{ACCESSION}"]),
            ]},
            {"search_filings": PASSAGES, "get_fact": FACT},
        )
        run = agent_loop.run_agent(PG_NET_INCOME.question, persist=False)
        assert run.evidence == [PASSAGES.summary, FACT.summary]
        assert run.draft_answer == ""

    def test_only_a_confidence_escalation_carries_a_draft(self, agent):
        agent({PG_NET_INCOME.question: [answer("Maybe $14.88 billion.", confidence=0.2)]})
        run = agent_loop.run_agent(PG_NET_INCOME.question, persist=False)
        assert run.outcome == "escalated"
        assert run.draft_answer == "Maybe $14.88 billion."
        assert run.answer.startswith("Low confidence (0.20)")


def test_migration_005_is_rerunnable():
    sql = pathlib.Path("sql/005_eval_result_agent.sql").read_text(encoding="utf-8")
    assert "ADD COLUMN IF NOT EXISTS agent JSONB" in sql
