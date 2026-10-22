"""Walk a workflow graph and run agent nodes through a harness."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from workbench.build import build_agent
from workbench.checks import findings_for, mask_pii
from workbench.document import HarnessSpec, Node, WorkflowDocument
from workbench.expressions import evaluate
from workbench.harness import DeltaCallback, HarnessResult, execute
from workbench.tools import file_search, mcp_stub

_FLOW_HANDLES = {"out", "pass", "true", "false", "body", "done", "approved", "rejected"}


@dataclass
class RunOutcome:
    status: str
    output: str
    events: list[dict[str, Any]] = field(default_factory=list)
    last_agent: str | None = None
    pending: dict[str, Any] | None = None
    error: str | None = None
    harness: str = "sync"
    deltas: str = ""
    report: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output": self.output,
            "events": self.events,
            "last_agent": self.last_agent,
            "pending": self.pending,
            "error": self.error,
            "harness": self.harness,
            "deltas": self.deltas,
            "report": self.report,
        }


async def arun_workflow(
    document: WorkflowDocument,
    user_input: str,
    *,
    harness: HarnessSpec | None = None,
    connectors: list[dict[str, Any]] | None = None,
    resume: dict[str, Any] | None = None,
    model_override: Any = None,
    on_delta: DeltaCallback | None = None,
    workflow_name: str | None = None,
) -> RunOutcome:
    spec = (harness or document.harness).model_copy(deep=True)
    state: dict[str, Any]
    node: Node | None
    events: list[dict[str, Any]] = []
    if resume:
        state = dict(resume.get("graph_state") or {})
        node = document.node(str(resume["node_id"]))
    else:
        state = {"input_as_text": user_input, "output": "", "vars": {}}
        node = document.start()
        events.append({"type": "node", "name": "start", "detail": "input_as_text", "node_id": node.id})
        node = document.successor(node.id, "out")

    last_agent: str | None = None
    deltas = ""
    steps = 0
    while node is not None and steps < 40:
        steps += 1
        events.append({"type": "node", "name": node.type, "detail": node.config.get("name") or node.id, "node_id": node.id})
        if node.type == "note":
            events.append({"type": "note", "name": node.id, "detail": str(node.config.get("text") or ""), "node_id": node.id})
            node = document.successor(node.id, "out")
            continue
        if node.type == "guardrails":
            blocked = _apply_guardrail(node, state, events)
            if blocked is not None:
                blocked.events = list(events)
                blocked.harness = spec.kind
                return blocked
            handle = "pass" if document.outgoing(node.id, "pass") else "out"
            node = document.successor(node.id, handle)
            continue
        if node.type == "if_else":
            try:
                taken = evaluate(str(node.config.get("expression") or ""), state)
            except ValueError as exc:
                return RunOutcome(status="failed", output="", events=events, error=str(exc), harness=spec.kind)
            events.append({"type": "branch", "name": node.id, "detail": "true" if taken else "false", "node_id": node.id})
            node = document.successor(node.id, "true" if taken else "false")
            continue
        if node.type == "while":
            key = f"__while_{node.id}"
            count = int(state["vars"].get(key) or 0)
            limit = int(node.config.get("max_iterations") or 3)
            try:
                again = count < limit and evaluate(str(node.config.get("expression") or "not_empty(input_as_text)"), state)
            except ValueError as exc:
                return RunOutcome(status="failed", output="", events=events, error=str(exc), harness=spec.kind)
            if again:
                body = document.successor(node.id, "body")
                if body is None:
                    node = document.successor(node.id, "done")
                    continue
                state["vars"][key] = count + 1
                events.append({"type": "loop", "name": node.id, "detail": f"iteration {count + 1}", "node_id": node.id})
                node = body
            else:
                node = document.successor(node.id, "done")
            continue
        if node.type == "human_approval":
            decision = _resume_decision(resume, node.id, "workflow")
            if decision is None:
                pending = {
                    "kind": "workflow",
                    "node_id": node.id,
                    "graph_state": _public_state(state),
                    "prompt": str(node.config.get("prompt") or "Approve this step?"),
                }
                return RunOutcome(
                    status="needs_approval",
                    output=str(state.get("output") or "Waiting for approval."),
                    events=events + [{"type": "approval", "name": node.id, "detail": pending["prompt"], "node_id": node.id}],
                    pending=pending,
                    harness=spec.kind,
                )
            resume = None
            events.append({"type": "approval", "name": node.id, "detail": decision, "node_id": node.id})
            if decision == "reject":
                rejected = document.successor(node.id, "rejected")
                if rejected is None:
                    return RunOutcome(
                        status="completed",
                        output="The reviewer rejected this step.",
                        events=events,
                        harness=spec.kind,
                    )
                node = rejected
                continue
            node = document.successor(node.id, "approved") or document.successor(node.id, "out")
            continue
        if node.type == "transform":
            state["output"] = _transform(node, state)
            node = document.successor(node.id, "out")
            continue
        if node.type == "set_state":
            key = str(node.config.get("key") or "value")
            source = str(node.config.get("source") or "output")
            state["vars"][key] = state["input_as_text"] if source == "input" else state.get("output") or ""
            events.append({"type": "state", "name": key, "detail": str(state["vars"][key])[:300], "node_id": node.id})
            node = document.successor(node.id, "out")
            continue
        if node.type == "file_search":
            query = str(state["vars"].get(node.config.get("query_var") or "query") or state.get("output") or state["input_as_text"])
            state["output"] = file_search(query)
            events.append({"type": "tool", "name": "file_search", "detail": query, "node_id": node.id})
            node = document.successor(node.id, "out")
            continue
        if node.type == "mcp":
            label = str(node.config.get("connector_id") or "connector")
            query = str(state.get("output") or state["input_as_text"])
            state["output"] = mcp_stub(label, query)
            events.append({"type": "tool", "name": label, "detail": "connector", "node_id": node.id})
            node = document.successor(node.id, "out")
            continue
        if node.type == "agent":
            try:
                agent = build_agent(node, document, spec, connectors=connectors)
            except Exception as exc:
                return RunOutcome(status="failed", output="", events=events, error=str(exc), harness=spec.kind)
            text_in = _agent_input(state)
            tool_resume = resume if _resume_decision(resume, node.id, "tool") is not None or (
                resume and resume.get("kind") == "tool" and resume.get("node_id") == node.id
            ) else None
            result = await execute(
                agent,
                text_in,
                spec,
                workflow_name=workflow_name or document.name,
                model_override=model_override,
                resume_state=(tool_resume or {}).get("sdk_state") if tool_resume else None,
                decision=(tool_resume or {}).get("decision") if tool_resume else None,
                on_delta=on_delta,
            )
            resume = None
            events.extend([{**event, "node_id": node.id} for event in result.events])
            deltas += result.deltas
            if result.last_agent:
                last_agent = result.last_agent
            if result.status == "needs_approval" and result.pending:
                pending = {
                    **result.pending,
                    "node_id": node.id,
                    "graph_state": _public_state(state),
                }
                return RunOutcome(
                    status="needs_approval",
                    output=result.output,
                    events=events,
                    last_agent=last_agent,
                    pending=pending,
                    harness=spec.kind,
                    deltas=deltas,
                )
            if result.status in {"failed", "blocked"}:
                return RunOutcome(
                    status=result.status,
                    output=result.output,
                    events=events,
                    last_agent=last_agent,
                    error=result.error,
                    harness=spec.kind,
                    deltas=deltas,
                )
            state["output"] = result.output
            node = document.successor(node.id, "out")
            continue
        if node.type == "start":
            node = document.successor(node.id, "out")
            continue
        return RunOutcome(status="failed", output="", events=events, error=f"Unsupported node {node.type}", harness=spec.kind)

    return RunOutcome(
        status="completed",
        output=str(state.get("output") or ""),
        events=events,
        last_agent=last_agent,
        harness=spec.kind,
        deltas=deltas,
    )


def run_workflow(
    document: WorkflowDocument,
    user_input: str,
    **kwargs: Any,
) -> RunOutcome:
    return asyncio.run(arun_workflow(document, user_input, **kwargs))


def _apply_guardrail(node: Node, state: dict[str, Any], events: list[dict[str, Any]]) -> RunOutcome | None:
    checks = [str(item) for item in node.config.get("checks") or ["pii", "jailbreak"]]
    action = str(node.config.get("action") or "flag")
    source = str(node.config.get("source") or "input")
    text = state["input_as_text"] if source == "input" else str(state.get("output") or "")
    found = findings_for(text, checks)
    events.append(
        {
            "type": "guardrail",
            "name": ",".join(checks),
            "detail": action if not found else ", ".join(item["detail"] for item in found),
            "node_id": node.id,
            "findings": found,
        }
    )
    if not found:
        return None
    if action == "mask":
        masked = mask_pii(text)
        if source == "input":
            state["input_as_text"] = masked
        else:
            state["output"] = masked
        return None
    detail = ", ".join(item["detail"] for item in found)
    return RunOutcome(
        status="blocked",
        output=f"Blocked by guardrails: {detail}.",
        events=[{"type": "guardrail", "name": "blocked", "detail": detail, "node_id": node.id}],
    )


def _transform(node: Node, state: dict[str, Any]) -> str:
    mode = str(node.config.get("mode") or "wrap")
    current = str(state.get("output") or "")
    if mode == "set":
        return str(node.config.get("value") or "")
    if mode == "append":
        return current + str(node.config.get("value") or "")
    if mode == "extract_json":
        import json

        try:
            payload = json.loads(current)
        except json.JSONDecodeError:
            return current
        key = str(node.config.get("key") or "answer")
        if isinstance(payload, dict) and key in payload:
            return str(payload[key])
        return current
    prefix = str(node.config.get("prefix") or "Result")
    return f"{prefix}: {current}"


def _agent_input(state: dict[str, Any]) -> str:
    query = state["vars"].get("query")
    if isinstance(query, str) and query.strip():
        return query
    output = str(state.get("output") or "").strip()
    if output:
        return output
    return str(state.get("input_as_text") or "")


def _resume_decision(resume: dict[str, Any] | None, node_id: str, kind: str) -> str | None:
    if not resume:
        return None
    if resume.get("node_id") == node_id and resume.get("kind") == kind:
        decision = resume.get("decision")
        return str(decision) if decision else None
    return None


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    variables = {key: value for key, value in state.get("vars", {}).items() if not str(key).startswith("__")}
    return {"input_as_text": state.get("input_as_text") or "", "output": state.get("output") or "", "vars": variables}


def outcome_from_harness(result: HarnessResult, spec: HarnessSpec) -> RunOutcome:
    return RunOutcome(
        status=result.status,
        output=result.output,
        events=result.events,
        last_agent=result.last_agent,
        pending=result.pending,
        error=result.error,
        harness=spec.kind,
        deltas=result.deltas,
    )
