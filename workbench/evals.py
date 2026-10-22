"""Datasets, graders, and trace grading.

Graders follow the dataset grader types that are safe to run locally:
string check, contains, token similarity, tool and handoff checks, guardrail
checks, and a word-count check. Model-graded scoring is left to a provider
that has an API key and is not invoked from this module.
"""

from __future__ import annotations

import re
from typing import Any

from workbench.document import EvalTask, WorkflowDocument
from workbench.runtime import RunOutcome, arun_workflow


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9']+", text.lower()) if token}


def similarity(left: str, right: str) -> float:
    a, b = tokenize(left), tokenize(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def grade_outcome(task: EvalTask, outcome: RunOutcome) -> dict[str, Any]:
    graders: list[dict[str, Any]] = []
    output = outcome.output or ""
    tool_names = [str(event.get("name")) for event in outcome.events if event.get("type") == "tool"]
    if task.expect_exact is not None:
        graders.append(_check("string_check", output.strip() == task.expect_exact.strip(), task.expect_exact))
    if task.expect_contains is not None:
        graders.append(
            _check("contains", task.expect_contains.lower() in output.lower(), task.expect_contains)
        )
    if task.expect_tool is not None:
        graders.append(_check("tool", task.expect_tool in tool_names, task.expect_tool))
    if task.expect_handoff is not None:
        blob = " ".join(
            str(event.get("name") or "")
            for event in outcome.events
            if event.get("type") in {"handoff", "agent"}
        )
        expected = task.expect_handoff.lower().replace(" ", "_")
        ok = outcome.last_agent == task.expect_handoff or expected in blob.lower()
        graders.append(_check("handoff", ok, task.expect_handoff))
    if task.expect_guardrail is not None:
        blocked = outcome.status == "blocked" or any(event.get("type") == "guardrail" and event.get("findings") for event in outcome.events)
        graders.append(_check("guardrail", blocked is task.expect_guardrail, str(task.expect_guardrail)))
    if task.expect_status is not None:
        graders.append(_check("status", outcome.status == task.expect_status, task.expect_status))
    if task.similarity_reference is not None:
        score = similarity(output, task.similarity_reference)
        graders.append(
            _check("similarity", score >= task.similarity_threshold, f"{score:.2f} >= {task.similarity_threshold}")
        )
    if task.max_words is not None:
        graders.append(_check("word_count", len(output.split()) <= task.max_words, f"<= {task.max_words}"))
    if not graders:
        graders.append(_check("completed", outcome.status == "completed", outcome.status))
    return {
        "input": task.input,
        "output": output,
        "status": outcome.status,
        "passed": all(item["passed"] for item in graders),
        "graders": graders,
        "last_agent": outcome.last_agent,
    }


def grade_trace(trace: dict[str, Any], task: EvalTask) -> dict[str, Any]:
    outcome = RunOutcome(
        status=str(trace.get("status") or ""),
        output=str(trace.get("output") or ""),
        events=list(trace.get("events") or []),
        last_agent=trace.get("last_agent"),
    )
    graded = grade_outcome(task, outcome)
    graded["trace_id"] = trace.get("id")
    return graded


async def run_eval(
    document: WorkflowDocument,
    tasks: list[EvalTask] | None = None,
    **kwargs: Any,
) -> RunOutcome:
    spec = document.harness.model_copy(deep=True)
    spec.session_backend = "none"
    spec.session_id = None
    if spec.kind == "eval":
        spec.kind = spec.eval_mode
    rows = []
    for task in tasks if tasks is not None else document.tasks:
        outcome = await arun_workflow(document, task.input, harness=spec, **kwargs)
        rows.append(grade_outcome(task, outcome))
    passed = sum(1 for row in rows if row["passed"])
    total = len(rows)
    return RunOutcome(
        status="completed" if rows else "failed",
        output=f"{passed}/{total} passed",
        events=[{"type": "eval", "name": row["input"][:80], "detail": "pass" if row["passed"] else "fail"} for row in rows],
        harness="eval",
        error=None if rows else "The dataset has no tasks.",
        report={"passed": passed, "failed": total - passed, "total": total, "results": rows},
    )


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}
