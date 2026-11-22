import asyncio

import pytest
from agents.tool import FunctionTool

from workbench.bench import (
    load_search_tasks,
    llm_correct_answer,
    parse_judge_payload,
    run_benchmark,
)
from workbench.build import build_agent
from workbench.cli import main
from workbench.document import EvalTask
from workbench.evals import grade_outcome, lexical_correct_answer
from workbench.export import export_python
from workbench.runtime import RunOutcome
from workbench.templates import get_template


def test_dataset_is_the_cookbook_slice() -> None:
    rows = load_search_tasks()
    assert len(rows) == 50
    assert rows[0]["id"] == "20caf138-0c81-4ef9-be60-fe919e0d68d4"
    assert rows[0]["question"] == "steve jobs statue location budapst"
    assert rows[0]["expected_answer"].startswith("The Steve Jobs statue is located in Budapest")
    assert rows[0]["category"] == "Arts"
    assert rows[0]["area"] == "Knowledge"
    assert rows[4]["question"] == "What event led to the creation of Google Images?"
    assert "Jennifer Lopez" in rows[4]["expected_answer"]
    assert len({row["id"] for row in rows}) == 50
    assert rows[5]["question"] == "how long does 503 service unavailable last on average"


def test_lexical_correct_answer_covers_the_sample_or_its_tokens() -> None:
    reference = "The Steve Jobs statue is located in Budapest, Hungary."
    score, recall = lexical_correct_answer(reference, reference)
    assert score == 1.0
    assert recall == 1.0

    score, _ = lexical_correct_answer("The year was 1956.", "1956")
    assert score == 1.0
    score, _ = lexical_correct_answer("No local passages matched the query.", "1956")
    assert score == 0.0

    covered = "Steve Jobs statue located Budapest"
    score, recall = lexical_correct_answer(covered, reference)
    assert score == 1.0
    assert recall >= 0.8

    weak = "Steve Jobs statue"
    score, recall = lexical_correct_answer(weak, reference)
    assert score == 0.0
    assert recall < 0.8


def test_reference_answer_grader_uses_correct_answer() -> None:
    task = EvalTask(input="Where is the statue?", reference_answer="Budapest, Hungary.")
    graded = grade_outcome(task, RunOutcome(status="completed", output="It is in Budapest, Hungary."))
    assert graded["passed"] is True
    grader = graded["graders"][0]
    assert grader["name"] == "correct_answer"
    assert grader["score"] == 1.0

    missed = grade_outcome(task, RunOutcome(status="completed", output="No local passages matched."))
    assert missed["passed"] is False
    assert missed["graders"][0]["score"] == 0.0


def test_template_matches_the_cookbook_agent() -> None:
    document = get_template("web_search_agent")
    node = document.node("web")
    assert node.config["name"] == "WebSearchAgent"
    assert node.config["instructions"] == "You are an agent that can search the web."
    assert node.config["tools"] == ["web_search"]
    assert node.config["search_context_size"] == "high"
    assert document.harness.temperature == 0.7

    document.harness.provider = "openai"
    hosted = build_agent(node, document, document.harness)
    assert hosted.name == "WebSearchAgent"
    assert hosted.tools[0].name == "web_search"
    assert type(hosted.tools[0]).__name__ == "WebSearchTool"
    assert hosted.tools[0].search_context_size == "high"

    local_doc = get_template("web_search_agent")
    local = build_agent(local_doc.node("web"), local_doc, local_doc.harness)
    assert isinstance(local.tools[0], FunctionTool)
    assert local.tools[0].name == "web_search"

    exported = export_python(get_template("web_search_agent"))
    assert 'WebSearchTool(search_context_size="high")' in exported
    assert "You are an agent that can search the web." in exported
    assert "function_tool" not in exported

    research = get_template("research")
    research.harness.provider = "openai"
    researcher = build_agent(research.node("researcher"), research, research.harness)
    web = next(tool for tool in researcher.tools if getattr(tool, "name", None) == "web_search")
    assert type(web).__name__ == "WebSearchTool"
    assert web.search_context_size == "medium"


def test_parse_judge_payload_accepts_binary_json() -> None:
    score, reason = parse_judge_payload('{"correct_answer": 1, "reason": "agrees"}')
    assert score == 1.0
    assert reason == "agrees"
    score, _ = parse_judge_payload('note {"correct_answer": 0, "reason": "misses the fact"} thanks')
    assert score == 0.0
    assert parse_judge_payload('{"correct_answer": true}')[0] == 1.0
    assert parse_judge_payload('{"correct_answer": false}')[0] == 0.0
    with pytest.raises(ValueError):
        parse_judge_payload('{"correct_answer": 0.5}')


def test_llm_judge_reads_json_and_usage_from_the_client() -> None:
    class Message:
        def __init__(self, content: str) -> None:
            self.content = content

    class Choice:
        def __init__(self, content: str) -> None:
            self.message = Message(content)

    class Details:
        cached_tokens = 3

    class Usage:
        prompt_tokens = 40
        completion_tokens = 9
        prompt_tokens_details = Details()

    class Response:
        def __init__(self, content: str) -> None:
            self.choices = [Choice(content)]
            self.usage = Usage()

    class Completions:
        def __init__(self) -> None:
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return Response('{"correct_answer": 1, "reason": "covered"}')

    class Chat:
        def __init__(self) -> None:
            self.completions = Completions()

    class Client:
        def __init__(self) -> None:
            self.chat = Chat()

    client = Client()
    score, reason, usage = llm_correct_answer("Where?", "Budapest", "Budapest, Hungary.", client=client)
    assert score == 1.0
    assert reason == "covered"
    assert usage == {
        "requests": 1,
        "input_tokens": 40,
        "cached_input_tokens": 3,
        "output_tokens": 9,
        "reasoning_tokens": 0,
    }
    sent = client.chat.completions.kwargs
    assert sent["model"] == "gpt-4o-mini"
    assert sent["temperature"] == 0
    assert "reasoning_effort" not in sent
    assert "Budapest" in sent["messages"][1]["content"]
    assert sent["messages"][0]["content"].startswith("You grade whether")


def test_local_benchmark_scores_every_cookbook_row() -> None:
    report = asyncio.run(run_benchmark("search_agent", provider="local"))
    assert report["provider"] == "local"
    assert report["judge"] == "local-overlap"
    assert report["model_scores"] is False
    assert report["total"] == 50
    assert report["errors"] == 0
    assert report["search_context_size"] == "high"
    assert report["search_context_applied"] is False
    assert report["results"][0]["correct_answer"] == 0.0
    assert report["results"][0]["tools"] == ["web_search"]
    assert report["results"][0]["last_agent"] == "WebSearchAgent"
    assert report["correct_answer_mean"] == report["correct"] / report["scored"]
    assert report["budget_usd"] == "2.000000"
    assert report["cost"]["usd"] == "0.000000"
    assert report["cost"]["web_search_usd"] == "0.000000"
    assert report["cost"]["web_search_calls"] == 50
    assert report["cost"]["budget_remaining_usd"] == "2.000000"
    assert report["skipped"] == 0
    assert report["results"][0]["cost"]["web_search_calls"] == 1
    for row in report["results"]:
        assert row["status"] == "completed"
        assert row["correct_answer"] in (0.0, 1.0)
        assert row["tools"] == ["web_search"]
        assert row["cost"]["usd"] == "0.000000"


def test_custom_judge_replaces_the_overlap_score() -> None:
    def judge(question: str, expected: str, output: str) -> tuple[float, str]:
        del question, expected, output
        return 1.0, "forced"

    report = asyncio.run(run_benchmark("search_agent", provider="local", limit=2, judge=judge))
    assert report["judge"] == "custom"
    assert report["model_scores"] is False
    assert report["correct_answer_mean"] == 1.0
    assert report["results"][0]["correct_answer"] == 1.0
    assert report["results"][0]["judge_reason"] == "forced"
    assert report["results"][0]["recall"] < 0.8


def test_openai_provider_uses_the_llm_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("workbench.bench.openai_api_key", lambda: "sk-test")

    async def fake_workflow(document, user_input, **kwargs):
        del user_input, kwargs
        assert document.harness.provider == "openai"
        assert document.harness.model == "gpt-4o"
        assert document.harness.temperature == 0.7
        assert document.node("web").config["search_context_size"] == "high"
        assert document.node("web").config["instructions"] == "You are an agent that can search the web."
        return RunOutcome(
            status="completed",
            output="stub answer",
            events=[{"type": "tool", "name": "web_search", "detail": "{}"}],
            last_agent="WebSearchAgent",
            usage={
                "requests": 2,
                "input_tokens": 100,
                "cached_input_tokens": 40,
                "output_tokens": 20,
            },
        )

    def fake_judge(question: str, expected: str, output: str, model: str | None = None):
        assert output == "stub answer"
        assert expected
        assert question
        assert model == "gpt-4o-mini"
        return 1.0, "matches the sample", {"requests": 1, "input_tokens": 30, "cached_input_tokens": 10, "output_tokens": 8}

    monkeypatch.setattr("workbench.bench.arun_workflow", fake_workflow)
    monkeypatch.setattr("workbench.bench.llm_correct_answer", fake_judge)
    report = asyncio.run(run_benchmark("search_agent", provider="openai", limit=2, budget_usd="2"))
    assert report["model_scores"] is True
    assert report["judge"] == "openai:gpt-4o-mini"
    assert report["model"] == "gpt-4o"
    assert report["search_context_applied"] is True
    assert report["correct_answer_mean"] == 1.0
    assert report["results"][0]["judge_reason"] == "matches the sample"
    assert report["results"][0]["cost"]["agent"]["input_tokens"] == 100
    assert report["results"][0]["cost"]["agent"]["cached_input_tokens"] == 40
    assert report["results"][0]["cost"]["judge"]["output_tokens"] == 8
    assert report["results"][0]["cost"]["web_search_calls"] == 1
    assert report["results"][0]["cost"]["web_search_usd"] == "0.010000"
    assert report["ran"] == 2
    assert report["skipped"] == 0


def test_budget_stops_before_the_next_task_and_never_spends_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("workbench.bench.openai_api_key", lambda: "sk-test")
    calls: list[str] = []

    async def fake_workflow(document, user_input, **kwargs):
        del document, kwargs
        calls.append(user_input)
        return RunOutcome(
            status="completed",
            output="priced answer",
            events=[{"type": "tool", "name": "web_search", "detail": "{}"}],
            last_agent="WebSearchAgent",
            usage={"requests": 1, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0},
        )

    def fake_judge(question: str, expected: str, output: str):
        del question, expected, output
        return 0.0, "no", {"requests": 1, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}

    monkeypatch.setattr("workbench.bench.arun_workflow", fake_workflow)
    from decimal import Decimal

    from workbench.pricing import estimate_task_usd

    estimate = estimate_task_usd(
        agent_model="gpt-4o",
        judge_model="gpt-4o-mini",
        max_turns=10,
        bill_web_search=True,
    )
    # One hosted web search is $0.01, under the reservation, so the second
    # task's reservation no longer fits in a budget equal to one estimate.
    report = asyncio.run(
        run_benchmark(
            "search_agent",
            provider="openai",
            limit=4,
            budget_usd=estimate,
            judge=fake_judge,
        )
    )
    assert calls and len(calls) == 1
    assert report["ran"] == 1
    assert report["skipped"] == 3
    assert [row["status"] for row in report["results"]] == ["completed", "skipped", "skipped", "skipped"]
    assert report["results"][1]["correct_answer"] is None
    assert "budget" in report["results"][1]["skipped_reason"]
    assert Decimal(report["cost"]["usd"]) == Decimal("0.010000")
    assert Decimal(report["cost"]["usd"]) <= Decimal(report["budget_usd"])
    assert report["results"][0]["cost"]["web_search_calls"] == 1
    assert report["cost"]["input_tokens"] == 0

    blocked = asyncio.run(
        run_benchmark(
            "search_agent",
            provider="openai",
            limit=3,
            budget_usd=0,
            judge=fake_judge,
        )
    )
    assert len(calls) == 1
    assert blocked["ran"] == 0
    assert blocked["skipped"] == 3
    assert blocked["cost"]["usd"] == "0.000000"
    assert all(row["status"] == "skipped" for row in blocked["results"])


def test_unpriced_model_is_refused_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("workbench.bench.openai_api_key", lambda: "sk-test")
    calls: list[str] = []

    async def fake_workflow(document, user_input, **kwargs):
        del document, kwargs
        calls.append(user_input)
        return RunOutcome(status="completed", output="x", last_agent="WebSearchAgent", usage={})

    monkeypatch.setattr("workbench.bench.arun_workflow", fake_workflow)
    with pytest.raises(ValueError, match="No price for model 'gpt-5.2'"):
        asyncio.run(
            run_benchmark(
                "search_agent",
                provider="openai",
                limit=3,
                model="gpt-5.2",
                judge_model_name="gpt-5-nano",
            )
        )
    assert calls == []


def test_model_access_error_aborts_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from workbench.bench import ModelAccessError

    monkeypatch.setattr("workbench.bench.openai_api_key", lambda: "sk-test")
    calls: list[int] = []

    async def denied(document, user_input, **kwargs):
        del document, user_input, kwargs
        calls.append(1)
        return RunOutcome(
            status="failed",
            output="",
            error="Error code: 403 - {'error': {'message': 'Project does not have access to model gpt-4o'}}",
            events=[{"type": "error", "name": "PermissionDeniedError", "detail": "Error code: 403"}],
            last_agent=None,
        )

    monkeypatch.setattr("workbench.bench.arun_workflow", denied)
    with pytest.raises(ModelAccessError, match="HTTP 403"):
        asyncio.run(run_benchmark("search_agent", provider="openai", limit=4, model="gpt-5-mini", judge_model_name="gpt-5-nano"))
    assert calls == [1]

    class Missing(Exception):
        status_code = 404

    async def ok(document, user_input, **kwargs):
        del document, user_input, kwargs
        calls.append(2)
        return RunOutcome(status="completed", output="answer", last_agent="WebSearchAgent", usage={})

    def missing_judge(question: str, expected: str, output: str):
        del question, expected, output
        raise Missing("model not found")

    monkeypatch.setattr("workbench.bench.arun_workflow", ok)
    with pytest.raises(ModelAccessError, match="HTTP 404"):
        asyncio.run(
            run_benchmark(
                "search_agent",
                provider="openai",
                limit=3,
                model="gpt-5-mini",
                judge_model_name="gpt-5-nano",
                judge=missing_judge,
            )
        )
    assert calls == [1, 2]


def test_gpt5_models_omit_temperature_and_set_low_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    from workbench.bench import judge_request_kwargs
    from workbench.document import HarnessSpec
    from workbench.provider import build_run_config

    nano = judge_request_kwargs("gpt-5-nano")
    assert nano == {"reasoning_effort": "low"}
    assert "temperature" not in nano
    assert judge_request_kwargs("gpt-4o-mini") == {"temperature": 0}

    monkeypatch.setattr("workbench.provider.openai_api_key", lambda: "sk-test")
    spec = HarnessSpec(provider="openai", model="gpt-5-mini", temperature=0.7, tracing=False)
    config = build_run_config(spec, workflow_name="bench")
    assert config.model == "gpt-5-mini"
    assert config.model_settings is not None
    assert config.model_settings.temperature is None
    assert config.model_settings.reasoning is not None
    assert config.model_settings.reasoning.effort == "low"

    classic = build_run_config(
        HarnessSpec(provider="openai", model="gpt-4o", temperature=0.2, tracing=False),
        workflow_name="bench",
    )
    assert classic.model_settings is not None
    assert classic.model_settings.temperature == 0.2
    assert classic.model_settings.reasoning is None


def test_openai_without_a_key_is_refused() -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        asyncio.run(run_benchmark("search_agent", provider="openai", limit=1))


def test_bench_cli_lists_and_runs_locally(tmp_path, capsys) -> None:
    assert main(["--home", str(tmp_path), "bench"]) == 0
    listed = capsys.readouterr().out
    assert "search_agent" in listed
    assert "evaluate_agents.ipynb" in listed

    assert main(["--home", str(tmp_path), "info"]) == 0
    assert "search_agent" in capsys.readouterr().out

    assert main(["--home", str(tmp_path), "bench", "missing"]) == 2
    assert "Unknown benchmark" in capsys.readouterr().err

    assert main(["--home", str(tmp_path), "bench", "search_agent", "--provider", "openai"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err

    assert main(["--home", str(tmp_path), "bench", "search_agent", "--limit", "2", "--json"]) == 0
    payload = capsys.readouterr().out
    assert '"benchmark": "search_agent"' in payload
    assert '"model_scores": false' in payload
    assert '"correct_answer": 0.0' in payload
    assert '"budget_usd": "2.000000"' in payload
    assert '"budget_remaining_usd"' in payload
