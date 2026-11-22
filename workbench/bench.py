"""Cookbook benchmarks.

`search_agent` reproduces the offline dataset evaluation in the OpenAI
Cookbook notebook ``examples/agents_sdk/evaluate_agents.ipynb``:

https://github.com/openai/openai-cookbook/blob/main/examples/agents_sdk/evaluate_agents.ipynb

That notebook runs an Agents SDK ``WebSearchAgent`` (``WebSearchTool``) on the
first 50 rows of Hugging Face ``junzhang1207/search-dataset`` and scores each
answer with a Langfuse LLM-as-a-judge named ``correct_answer``. The judge
compares the agent's answer to the dataset sample answer. The write-up reports
mean correctness of about 0.89 to 0.92 as ``search_context_size`` increases.
This runner uses the ``high`` setting from that comparison. The vendored file
is the first 50 rows of the current ``data.jsonl`` (929 rows on the dataset
card; the notebook log printed 934).

Langfuse tracing is not reproduced. The task, the 50-row slice, the agent, and
the ``correct_answer`` score are. The notebook does not include the judge
prompt (it is configured in the Langfuse UI). The prompt here implements that
described check and uses ``gpt-4o-mini``, the judge model in the notebook's
LLM-as-a-judge section. Without an API key, a deterministic token-recall
stand-in produces the same score name so the pipeline can run offline.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from workbench.document import EvalTask, HarnessSpec
from workbench.env import openai_api_key
from workbench.evals import grade_outcome
from workbench.pricing import (
    PRICING_CHECKED,
    PRICING_SOURCE,
    RATES,
    WEB_SEARCH_USD_PER_CALL,
    add_usage,
    can_afford,
    empty_usage,
    estimate_task_usd,
    money,
    price_tokens,
    price_web_search,
    usage_from_chat_completion,
)
from workbench.runtime import arun_workflow
from workbench.templates import get_template

COOKBOOK_URL = (
    "https://github.com/openai/openai-cookbook/blob/main/examples/agents_sdk/evaluate_agents.ipynb"
)
DATASET_NAME = "junzhang1207/search-dataset"
# Current data.jsonl / dataset card. The notebook's saved output printed 934.
DATASET_TOTAL = 929
JUDGE_MODEL = "gpt-4o-mini"
COOKBOOK_AGENT_MODEL = "gpt-4o"

# (score, reason) or (score, reason, usage dict).
Judge = Callable[..., tuple]
DEFAULT_BUDGET_USD = Decimal("2.00")

BENCHMARKS: dict[str, dict[str, str]] = {
    "search_agent": {
        "id": "search_agent",
        "template": "web_search_agent",
        "summary": "WebSearchAgent answer correctness on the cookbook search-dataset slice.",
        "source": COOKBOOK_URL,
    }
}


def list_benchmarks() -> list[dict[str, str]]:
    return [dict(item) for item in BENCHMARKS.values()]


def dataset_path() -> Path:
    packaged = Path(__file__).resolve().parent.parent / "benchmarks" / "search_agent" / "tasks.jsonl"
    if packaged.exists():
        return packaged
    return Path.cwd() / "benchmarks" / "search_agent" / "tasks.jsonl"


def load_search_tasks(limit: int | None = None) -> list[dict[str, str]]:
    path = dataset_path()
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {path}")
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows.append(
            {
                "id": str(item["id"]),
                "question": str(item["question"]),
                "expected_answer": str(item["expected_answer"]),
                "category": str(item.get("category") or ""),
                "area": str(item.get("area") or ""),
            }
        )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        return rows[:limit]
    return rows


def resolve_provider(requested: str | None) -> str:
    choice = (requested or "auto").strip().lower()
    if choice in {"", "auto"}:
        return "openai" if openai_api_key() else "local"
    if choice not in {"local", "openai"}:
        raise ValueError(f"Unknown provider {requested!r}. Use auto, local, or openai.")
    if choice == "openai" and not openai_api_key():
        raise RuntimeError(
            "provider is openai but OPENAI_API_KEY is not set. "
            "Export the key, or run: python -m workbench bench search_agent --provider local"
        )
    return choice


def judge_model() -> str:
    return os.environ.get("OPENAI_JUDGE_MODEL", JUDGE_MODEL).strip() or JUDGE_MODEL


def agent_model(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get("OPENAI_MODEL", "").strip()
    return env or COOKBOOK_AGENT_MODEL


JUDGE_SYSTEM = (
    "You grade whether an agent's answer is correct given a sample answer. "
    'Reply with JSON only: {"correct_answer": 0 or 1, "reason": "one sentence"}. '
    "Score 1 when the agent answer agrees with the sample answer on the facts the question asks for. "
    "That includes a sample answer which says the question has a false premise. "
    "Score 0 when the agent answer misses that fact, contradicts the sample, or does not answer. "
    "Wording may differ. Extra detail is fine when it does not contradict the sample."
)


def judge_user_prompt(question: str, expected: str, output: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Sample answer:\n{expected}\n\n"
        f"Agent answer:\n{output}\n"
    )


def parse_judge_payload(text: str) -> tuple[float, str]:
    raw = text.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Judge did not return JSON: {raw[:200]}")
        data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict) or "correct_answer" not in data:
        raise ValueError(f"Judge JSON is missing correct_answer: {raw[:200]}")
    score = data["correct_answer"]
    if isinstance(score, bool):
        score = int(score)
    if score not in (0, 1, 0.0, 1.0):
        raise ValueError(f"correct_answer must be 0 or 1, got {score!r}")
    reason = str(data.get("reason") or "").strip()
    return float(score), reason


def llm_correct_answer(
    question: str,
    expected: str,
    output: str,
    *,
    client: Any | None = None,
) -> tuple[float, str, dict[str, int]]:
    """Call the cookbook-style correct_answer judge. Requires an API key."""
    if client is None:
        key = openai_api_key()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set, so the correct_answer judge cannot run.")
        from openai import OpenAI

        client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=judge_model(),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": judge_user_prompt(question, expected, output)},
        ],
    )
    text = response.choices[0].message.content or ""
    score, reason = parse_judge_payload(text)
    return score, reason, usage_from_chat_completion(getattr(response, "usage", None))


async def run_benchmark(
    name: str,
    *,
    provider: str | None = "auto",
    limit: int | None = None,
    model: str | None = None,
    context_size: str = "high",
    judge: Judge | None = None,
    budget_usd: Decimal | float | str = DEFAULT_BUDGET_USD,
) -> dict[str, Any]:
    if name not in BENCHMARKS:
        known = ", ".join(BENCHMARKS)
        raise KeyError(f"Unknown benchmark {name!r}. Known benchmarks: {known}")
    if context_size not in {"low", "medium", "high"}:
        raise ValueError(f"context_size must be low, medium, or high, got {context_size!r}")
    resolved = resolve_provider(provider)
    items = load_search_tasks(limit)
    document = get_template(BENCHMARKS[name]["template"])
    document.harness.provider = resolved  # type: ignore[assignment]
    document.node("web").config["search_context_size"] = context_size
    chosen_model = "local"
    judge_name = "local-overlap"
    if resolved == "openai":
        chosen_model = agent_model(model)
        document.harness.model = chosen_model
        if judge is None:
            judge = llm_correct_answer
            judge_name = f"openai:{judge_model()}"
        else:
            judge_name = "custom"
    elif judge is not None:
        judge_name = "custom"

    budget = _budget(budget_usd)
    bill_search = resolved == "openai"
    priced_judge = judge_model() if judge is not None and bill_search else None
    estimate = estimate_task_usd(
        agent_model=chosen_model,
        judge_model=priced_judge,
        max_turns=document.harness.max_turns,
        bill_web_search=bill_search,
    )
    spec = _eval_spec(document.harness)
    spent = Decimal(0)
    agent_totals = empty_usage()
    judge_totals = empty_usage()
    agent_usd_total = Decimal(0)
    judge_usd_total = Decimal(0)
    search_calls = 0
    search_usd = Decimal(0)
    rows: list[dict[str, Any]] = []
    errors = 0
    skipped = 0
    stop_for_budget = False
    for item in items:
        task = EvalTask(input=item["question"], reference_answer=item["expected_answer"])
        if stop_for_budget or not can_afford(spent, estimate, budget):
            stop_for_budget = True
            skipped += 1
            rows.append(_skipped_row(item, budget, spent, estimate))
            continue
        outcome = await arun_workflow(document, item["question"], harness=spec)
        graded = grade_outcome(task, outcome)
        agent_usage = dict(outcome.usage or empty_usage())
        calls = sum(1 for name in graded.get("tools") or [] if name == "web_search")
        agent_usd = Decimal(0) if chosen_model == "local" else price_tokens(chosen_model, **_token_kwargs(agent_usage))
        call_usd = price_web_search(calls, billable=bill_search)
        judge_usage = empty_usage()
        judge_usd = Decimal(0)
        score, recall, reason, error = _score_from_grade(graded)
        if judge is not None:
            try:
                judged_score, reason, judge_usage = _judge_parts(
                    judge(item["question"], item["expected_answer"], str(graded.get("output") or ""))
                )
            except Exception as exc:
                error = str(exc)
                errors += 1
                score = None
            else:
                score = judged_score
                if priced_judge:
                    judge_usd = price_tokens(priced_judge, **_token_kwargs(judge_usage))
        elif error is None and score is None:
            error = "The eval harness did not return a correct_answer score."
            errors += 1
        task_usd = agent_usd + call_usd + judge_usd
        spent += task_usd
        agent_totals = add_usage(agent_totals, agent_usage)
        judge_totals = add_usage(judge_totals, judge_usage)
        agent_usd_total += agent_usd
        judge_usd_total += judge_usd
        search_calls += calls
        search_usd += call_usd
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "area": item["area"],
                "input": item["question"],
                "expected_answer": item["expected_answer"],
                "output": graded.get("output") or "",
                "status": graded.get("status"),
                "correct_answer": score,
                "recall": recall,
                "judge_reason": reason,
                "passed": score == 1.0,
                "tools": list(graded.get("tools") or []),
                "last_agent": graded.get("last_agent"),
                "error": error,
                "cost": _cost_block(
                    usd=task_usd,
                    usage=add_usage(agent_usage, judge_usage),
                    web_search_calls=calls,
                    web_search_usd=call_usd,
                    agent_model=chosen_model,
                    agent_usage=agent_usage,
                    agent_usd=agent_usd,
                    judge_model=priced_judge or judge_name,
                    judge_usage=judge_usage,
                    judge_usd=judge_usd,
                    budget_remaining=budget - spent,
                ),
            }
        )
    scored_rows = [row for row in rows if isinstance(row["correct_answer"], (int, float))]
    correct = sum(1 for row in scored_rows if row["correct_answer"] == 1.0)
    mean = (sum(float(row["correct_answer"]) for row in scored_rows) / len(scored_rows)) if scored_rows else None
    model_scores = resolved == "openai" and judge_name.startswith("openai:")
    ran = len(rows) - skipped
    return {
        "benchmark": name,
        "source": COOKBOOK_URL,
        "dataset": DATASET_NAME,
        "dataset_total": DATASET_TOTAL,
        "dataset_rows": len(items),
        "dataset_note": (
            f"first {len(items)} of {DATASET_TOTAL} rows of {DATASET_NAME} "
            "(the slice the cookbook uploads)"
            if limit is None
            else f"first {len(items)} rows of the cookbook's 50-row slice of {DATASET_NAME}"
        ),
        "provider": resolved,
        "model": chosen_model,
        "judge": judge_name,
        "model_scores": model_scores,
        "search_context_size": context_size,
        "search_context_applied": bill_search,
        "temperature": document.harness.temperature,
        "correct_answer_mean": mean,
        "correct": correct,
        "scored": len(scored_rows),
        "ran": ran,
        "skipped": skipped,
        "total": len(rows),
        "errors": errors,
        "budget_usd": money(budget),
        "estimate_usd_per_task": money(estimate),
        "pricing": {
            "source": PRICING_SOURCE,
            "checked": PRICING_CHECKED,
            "web_search_usd_per_call": money(WEB_SEARCH_USD_PER_CALL),
            "per_million_tokens": {
                name: {
                    "input": money(rates.input),
                    "cached_input": money(rates.cached_input),
                    "output": money(rates.output),
                }
                for name, rates in RATES.items()
            },
        },
        "cost": _cost_block(
            usd=spent,
            usage=add_usage(agent_totals, judge_totals),
            web_search_calls=search_calls,
            web_search_usd=search_usd,
            agent_model=chosen_model,
            agent_usage=agent_totals,
            agent_usd=agent_usd_total,
            judge_model=priced_judge or judge_name,
            judge_usage=judge_totals,
            judge_usd=judge_usd_total,
            budget_remaining=budget - spent,
        ),
        "results": rows,
    }


def _budget(value: Decimal | float | str) -> Decimal:
    try:
        budget = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"budget_usd must be a number, got {value!r}") from exc
    if budget < 0:
        raise ValueError("budget_usd must be >= 0")
    return budget


def _eval_spec(harness: HarnessSpec) -> HarnessSpec:
    spec = harness.model_copy(deep=True)
    spec.session_backend = "none"
    spec.session_id = None
    if spec.kind == "eval":
        spec.kind = spec.eval_mode
    return spec


def _token_kwargs(usage: dict[str, int]) -> dict[str, int]:
    return {
        "input_tokens": int(usage.get("input_tokens", 0)),
        "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
    }


def _judge_parts(value: Any) -> tuple[float, str, dict[str, int]]:
    if not isinstance(value, tuple) or len(value) < 2:
        raise TypeError("Judge must return (score, reason) or (score, reason, usage)")
    usage = value[2] if len(value) >= 3 and isinstance(value[2], dict) else empty_usage()
    return float(value[0]), str(value[1]), {**empty_usage(), **usage}


def _score_from_grade(graded: dict[str, Any]) -> tuple[float | None, float | None, str, str | None]:
    lexical = _grader(graded, "correct_answer")
    if lexical is None:
        return None, None, "", None
    return lexical.get("score"), lexical.get("recall"), "", None


def _skipped_row(item: dict[str, str], budget: Decimal, spent: Decimal, estimate: Decimal) -> dict[str, Any]:
    reason = (
        f"Stopped before this task: spent {money(spent)} USD plus estimate {money(estimate)} USD "
        f"would exceed budget {money(budget)} USD."
    )
    return {
        "id": item["id"],
        "category": item["category"],
        "area": item["area"],
        "input": item["question"],
        "expected_answer": item["expected_answer"],
        "output": "",
        "status": "skipped",
        "correct_answer": None,
        "recall": None,
        "judge_reason": "",
        "passed": False,
        "tools": [],
        "last_agent": None,
        "error": None,
        "skipped_reason": reason,
        "cost": _cost_block(
            usd=Decimal(0),
            usage=empty_usage(),
            web_search_calls=0,
            web_search_usd=Decimal(0),
            agent_model="",
            agent_usage=empty_usage(),
            agent_usd=Decimal(0),
            judge_model="",
            judge_usage=empty_usage(),
            judge_usd=Decimal(0),
            budget_remaining=budget - spent,
        ),
    }


def _cost_block(
    *,
    usd: Decimal,
    usage: dict[str, int],
    web_search_calls: int,
    web_search_usd: Decimal,
    agent_model: str,
    agent_usage: dict[str, int],
    agent_usd: Decimal,
    judge_model: str,
    judge_usage: dict[str, int],
    judge_usd: Decimal,
    budget_remaining: Decimal,
) -> dict[str, Any]:
    return {
        "usd": money(usd),
        "budget_remaining_usd": money(budget_remaining),
        "input_tokens": int(usage.get("input_tokens", 0)),
        "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "requests": int(usage.get("requests", 0)),
        "web_search_calls": web_search_calls,
        "web_search_usd": money(web_search_usd),
        "agent": _party_cost(agent_model, agent_usage, agent_usd),
        "judge": _party_cost(judge_model, judge_usage, judge_usd),
    }


def _party_cost(model: str, usage: dict[str, int], usd: Decimal) -> dict[str, Any]:
    return {
        "model": model,
        "requests": int(usage.get("requests", 0)),
        "input_tokens": int(usage.get("input_tokens", 0)),
        "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "usd": money(usd),
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"{report['benchmark']}  provider={report['provider']}  model={report['model']}  judge={report['judge']}",
        f"source: {report['source']}",
        f"dataset: {report['dataset_note']}",
        (
            f"search_context_size={report['search_context_size']}  "
            f"hosted_tool={'yes' if report['search_context_applied'] else 'no'}"
        ),
    ]
    if not report["model_scores"]:
        lines.append(
            "Scores are from the local deterministic provider and an overlap judge. They are not model scores."
        )
    lines.append("")
    for index, row in enumerate(report["results"]):
        score = row["correct_answer"]
        if row.get("status") == "skipped":
            shown = "skipped"
        elif score is None:
            shown = "error"
        else:
            shown = f"{score:.0f}"
        recall = row.get("recall")
        recall_text = "n/a" if not isinstance(recall, (int, float)) else f"{recall:.2f}"
        tools = ",".join(row.get("tools") or []) or "-"
        lines.append(
            f"[{index}] {row['category']}/{row['area']}  correct_answer={shown}  "
            f"recall={recall_text}  tools={tools}"
        )
        lines.append(f"    Q: {row['input']}")
        if row.get("status") != "skipped":
            output = " ".join(str(row.get("output") or "").split())
            if len(output) > 180:
                output = output[:177] + "..."
            lines.append(f"    A: {output}")
        cost = row.get("cost") or {}
        lines.append(
            f"    cost {cost.get('usd', '0.000000')} USD  "
            f"in={cost.get('input_tokens', 0)} cached={cost.get('cached_input_tokens', 0)} "
            f"out={cost.get('output_tokens', 0)}  web_search={cost.get('web_search_calls', 0)}  "
            f"budget_remaining={cost.get('budget_remaining_usd', report.get('budget_usd'))} USD"
        )
        if row.get("judge_reason"):
            lines.append(f"    judge: {row['judge_reason']}")
        if row.get("skipped_reason"):
            lines.append(f"    skipped: {row['skipped_reason']}")
        if row.get("error"):
            lines.append(f"    error: {row['error']}")
    mean = report["correct_answer_mean"]
    mean_text = "n/a" if mean is None else f"{mean:.2f}"
    cost = report.get("cost") or {}
    lines.append("")
    lines.append(
        f"correct_answer mean: {mean_text} "
        f"({report['correct']}/{report['scored']} correct, {report['ran']} ran, "
        f"{report.get('skipped', 0)} skipped, {report['total']} tasks)"
    )
    lines.append(
        f"cost {cost.get('usd', '0.000000')} USD  "
        f"in={cost.get('input_tokens', 0)} cached={cost.get('cached_input_tokens', 0)} "
        f"out={cost.get('output_tokens', 0)}  web_search={cost.get('web_search_calls', 0)} "
        f"({cost.get('web_search_usd', '0.000000')} USD)"
    )
    lines.append(
        f"budget {report.get('budget_usd')} USD  "
        f"remaining {cost.get('budget_remaining_usd')} USD  "
        f"estimate_per_task {report.get('estimate_usd_per_task')} USD"
    )
    if not report["model_scores"]:
        lines.append("These scores are not real model scores.")
        lines.append("Run against OpenAI with a budget cap:")
        lines.append(
            "  OPENAI_API_KEY=... python -m workbench bench search_agent --provider openai --budget-usd 2 --limit 5"
        )
    return "\n".join(lines)


def _grader(result: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in result.get("graders") or []:
        if row.get("name") == name:
            return row
    return None
