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
from pathlib import Path
from typing import Any, Callable

from workbench.document import EvalTask
from workbench.env import openai_api_key
from workbench.evals import run_eval
from workbench.templates import get_template

COOKBOOK_URL = (
    "https://github.com/openai/openai-cookbook/blob/main/examples/agents_sdk/evaluate_agents.ipynb"
)
DATASET_NAME = "junzhang1207/search-dataset"
# Current data.jsonl / dataset card. The notebook's saved output printed 934.
DATASET_TOTAL = 929
JUDGE_MODEL = "gpt-4o-mini"
COOKBOOK_AGENT_MODEL = "gpt-4o"

Judge = Callable[[str, str, str], tuple[float, str]]

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
) -> tuple[float, str]:
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
    return parse_judge_payload(text)


async def run_benchmark(
    name: str,
    *,
    provider: str | None = "auto",
    limit: int | None = None,
    model: str | None = None,
    context_size: str = "high",
    judge: Judge | None = None,
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

    tasks = [EvalTask(input=item["question"], reference_answer=item["expected_answer"]) for item in items]
    outcome = await run_eval(document, tasks)
    results = list((outcome.report or {}).get("results") or [])
    if len(results) != len(items):
        raise RuntimeError(f"Eval returned {len(results)} rows for {len(items)} tasks.")
    rows: list[dict[str, Any]] = []
    errors = 0
    for item, result in zip(items, results):
        lexical = _grader(result, "correct_answer")
        score = None if lexical is None else lexical.get("score")
        recall = None if lexical is None else lexical.get("recall")
        reason = ""
        error = None
        if judge is not None:
            try:
                score, reason = judge(item["question"], item["expected_answer"], str(result.get("output") or ""))
            except Exception as exc:
                error = str(exc)
                errors += 1
                score = None
            else:
                graders = [row for row in result.get("graders") or [] if row.get("name") != "correct_answer"]
                graders.append(
                    {
                        "name": "correct_answer",
                        "passed": score == 1.0,
                        "detail": reason or f"{score:.0f}",
                        "score": score,
                        "recall": recall,
                    }
                )
                result["graders"] = graders
                result["passed"] = all(row.get("passed") for row in graders)
        if error is None and score is None:
            error = "The eval harness did not return a correct_answer score."
            errors += 1
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "area": item["area"],
                "input": item["question"],
                "expected_answer": item["expected_answer"],
                "output": result.get("output") or "",
                "status": result.get("status"),
                "correct_answer": score,
                "recall": recall,
                "judge_reason": reason,
                "passed": score == 1.0,
                "tools": list(result.get("tools") or []),
                "last_agent": result.get("last_agent"),
                "error": error,
            }
        )
    scored = [row for row in rows if isinstance(row["correct_answer"], (int, float))]
    correct = sum(1 for row in scored if row["correct_answer"] == 1.0)
    mean = (sum(float(row["correct_answer"]) for row in scored) / len(scored)) if scored else None
    model_scores = resolved == "openai" and judge_name.startswith("openai:")
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
        "search_context_applied": resolved == "openai",
        "temperature": document.harness.temperature,
        "correct_answer_mean": mean,
        "correct": correct,
        "scored": len(scored),
        "total": len(rows),
        "errors": errors,
        "results": rows,
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
        shown = "error" if score is None else f"{score:.0f}"
        recall = row.get("recall")
        recall_text = "n/a" if not isinstance(recall, (int, float)) else f"{recall:.2f}"
        tools = ",".join(row.get("tools") or []) or "-"
        lines.append(
            f"[{index}] {row['category']}/{row['area']}  correct_answer={shown}  "
            f"recall={recall_text}  tools={tools}"
        )
        lines.append(f"    Q: {row['input']}")
        output = " ".join(str(row.get("output") or "").split())
        if len(output) > 180:
            output = output[:177] + "..."
        lines.append(f"    A: {output}")
        if row.get("judge_reason"):
            lines.append(f"    judge: {row['judge_reason']}")
        if row.get("error"):
            lines.append(f"    error: {row['error']}")
    mean = report["correct_answer_mean"]
    mean_text = "n/a" if mean is None else f"{mean:.2f}"
    lines.append("")
    lines.append(
        f"correct_answer mean: {mean_text} "
        f"({report['correct']}/{report['scored']} correct, {report['total']} tasks)"
    )
    if not report["model_scores"]:
        lines.append("These scores are not real model scores.")
        lines.append("Run against OpenAI with:")
        lines.append("  OPENAI_API_KEY=... python -m workbench bench search_agent --provider openai")
    return "\n".join(lines)


def _grader(result: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in result.get("graders") or []:
        if row.get("name") == name:
            return row
    return None
