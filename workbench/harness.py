"""Pluggable runtimes around `Runner`.

A harness chooses sync, async, or streaming execution and applies shared run
options: max turns, tracing, model provider, model settings, session memory,
and human approval of tool calls. Register a new harness with `register_harness`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    OutputGuardrailTripwireTriggered,
    Runner,
    RunState,
    SQLiteSession,
)
from agents.stream_events import RawResponsesStreamEvent

from workbench.document import HarnessSpec
from workbench.paths import workbench_home
from workbench.provider import build_run_config

DeltaCallback = Callable[[str], Awaitable[None]]
_MEMORY_SESSIONS: dict[str, SQLiteSession] = {}


@dataclass
class HarnessResult:
    status: str
    output: str
    events: list[dict[str, Any]] = field(default_factory=list)
    last_agent: str | None = None
    pending: dict[str, Any] | None = None
    error: str | None = None
    deltas: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Harness:
    """Base class. Subclasses pick how a single `Runner` invocation is driven."""

    name = "sync"

    async def arun(
        self,
        agent: Agent[Any],
        user_input: str,
        spec: HarnessSpec,
        *,
        workflow_name: str = "agent-workbench",
        model_override: Any = None,
        resume_state: str | None = None,
        decision: str | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> HarnessResult:
        raise NotImplementedError

    def run(self, agent: Agent[Any], user_input: str, spec: HarnessSpec, **kwargs: Any) -> HarnessResult:
        return asyncio.run(self.arun(agent, user_input, spec, **kwargs))


class SyncHarness(Harness):
    name = "sync"


class AsyncHarness(Harness):
    name = "async"


class StreamingHarness(Harness):
    name = "streaming"


class ApprovalHarness(Harness):
    """Tool-approval policy harness. Execution itself is sync unless streaming is set."""

    name = "approval"


class EvalHarness(Harness):
    """Single-task execution used inside an eval sweep. The sweep lives in `evals`."""

    name = "eval"


HARNESSES: dict[str, Harness] = {
    "sync": SyncHarness(),
    "async": AsyncHarness(),
    "streaming": StreamingHarness(),
    "approval": ApprovalHarness(),
    "eval": EvalHarness(),
}


def register_harness(harness: Harness) -> None:
    HARNESSES[harness.name] = harness


def get_harness(kind: str) -> Harness:
    try:
        return HARNESSES[kind]
    except KeyError as exc:
        known = ", ".join(sorted(HARNESSES))
        raise KeyError(f"Unknown harness {kind!r}. Known harnesses: {known}") from exc


async def execute(
    agent: Agent[Any],
    user_input: str,
    spec: HarnessSpec,
    *,
    workflow_name: str = "agent-workbench",
    model_override: Any = None,
    resume_state: str | None = None,
    decision: str | None = None,
    on_delta: DeltaCallback | None = None,
) -> HarnessResult:
    """Run one agent with the shared SDK loop and the harness's execution mode."""
    harness = get_harness(spec.kind)
    if harness.__class__.arun is not Harness.arun:
        return await harness.arun(
            agent,
            user_input,
            spec,
            workflow_name=workflow_name,
            model_override=model_override,
            resume_state=resume_state,
            decision=decision,
            on_delta=on_delta,
        )
    mode = spec.eval_mode if spec.kind == "eval" else harness.name
    if mode == "approval":
        mode = "sync"
    deltas: list[str] = []

    async def collect(text: str) -> None:
        deltas.append(text)
        if on_delta is not None:
            await on_delta(text)

    try:
        run_config = build_run_config(spec, workflow_name=workflow_name, model_override=model_override)
        session = open_session(spec)
        if resume_state:
            state = await RunState.from_string(agent, resume_state)
            for item in state.get_interruptions():
                if decision == "reject":
                    state.reject(item, rejection_message="The reviewer rejected this tool call.")
                else:
                    state.approve(item)
            result = await Runner.run(
                agent,
                state,
                max_turns=spec.max_turns,
                run_config=run_config,
                session=session,
            )
        elif mode == "streaming":
            result = Runner.run_streamed(
                agent,
                user_input,
                max_turns=spec.max_turns,
                run_config=run_config,
                session=session,
            )
            async for event in result.stream_events():
                if isinstance(event, RawResponsesStreamEvent):
                    data = event.data
                    if getattr(data, "type", None) == "response.output_text.delta":
                        await collect(str(getattr(data, "delta", "")))
        else:
            result = await Runner.run(
                agent,
                user_input,
                max_turns=spec.max_turns,
                run_config=run_config,
                session=session,
            )
    except InputGuardrailTripwireTriggered as exc:
        return _blocked("input", exc)
    except OutputGuardrailTripwireTriggered as exc:
        return _blocked("output", exc)
    except MaxTurnsExceeded as exc:
        return HarnessResult(status="failed", output="", error=str(exc), events=[{"type": "error", "name": "max_turns", "detail": str(exc)}])
    except Exception as exc:
        return HarnessResult(status="failed", output="", error=str(exc), events=[{"type": "error", "name": type(exc).__name__, "detail": str(exc)}])

    if getattr(result, "interruptions", None):
        if spec.approval == "auto":
            state = result.to_state()
            for item in result.interruptions:
                state.approve(item)
            try:
                result = await Runner.run(agent, state, max_turns=spec.max_turns, run_config=run_config, session=session)
            except Exception as exc:
                return HarnessResult(status="failed", output="", error=str(exc))
        elif spec.approval == "reject":
            state = result.to_state()
            for item in result.interruptions:
                state.reject(item, rejection_message="Rejected by the approval harness.")
            try:
                result = await Runner.run(agent, state, max_turns=spec.max_turns, run_config=run_config, session=session)
            except Exception as exc:
                return HarnessResult(status="failed", output="", error=str(exc))
        else:
            pending_state = result.to_state()
            return HarnessResult(
                status="needs_approval",
                output="Waiting for approval before a tool runs.",
                events=_events(result) + [
                    {"type": "approval", "name": item.tool_name or "tool", "detail": "pending"}
                    for item in result.interruptions
                ],
                last_agent=result.last_agent.name if result.last_agent else None,
                pending={
                    "kind": "tool",
                    "sdk_state": pending_state.to_string(),
                    "tools": [item.tool_name for item in result.interruptions],
                    "prompt": "Approve this tool call?",
                },
                deltas="".join(deltas),
            )

    output = _stringify(result.final_output)
    return HarnessResult(
        status="completed",
        output=output,
        events=_events(result),
        last_agent=result.last_agent.name if result.last_agent else agent.name,
        deltas="".join(deltas),
    )


def run_agent(agent: Agent[Any], user_input: str, spec: HarnessSpec | None = None, **kwargs: Any) -> HarnessResult:
    """Synchronous helper for custom agents built in code."""
    return asyncio.run(execute(agent, user_input, spec or HarnessSpec(), **kwargs))


def open_session(spec: HarnessSpec) -> SQLiteSession | None:
    if spec.session_backend == "none" or not spec.session_id:
        return None
    if spec.session_backend == "memory":
        session = _MEMORY_SESSIONS.get(spec.session_id)
        if session is None:
            session = SQLiteSession(spec.session_id)
            _MEMORY_SESSIONS[spec.session_id] = session
        return session
    return SQLiteSession(spec.session_id, workbench_home() / "sessions.db")


def clear_memory_sessions() -> None:
    _MEMORY_SESSIONS.clear()


def _blocked(where: str, exc: InputGuardrailTripwireTriggered | OutputGuardrailTripwireTriggered) -> HarnessResult:
    info = getattr(exc.guardrail_result.output, "output_info", {}) or {}
    findings = info.get("findings") or []
    detail = ", ".join(item.get("detail", item.get("check", "")) for item in findings) or str(exc)
    return HarnessResult(
        status="blocked",
        output=f"Blocked by the {where} guardrail: {detail}.",
        events=[{"type": "guardrail", "name": where, "detail": detail, "findings": findings}],
        error=None,
    )


def _events(result: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for guardrail in getattr(result, "input_guardrail_results", []) or []:
        output = guardrail.output
        if output.tripwire_triggered or (output.output_info or {}).get("findings"):
            events.append({"type": "guardrail", "name": "input", "detail": str(output.output_info)})
    for item in result.new_items:
        kind = getattr(item, "type", "")
        if kind == "tool_call_item":
            raw = item.raw_item
            events.append(
                {
                    "type": "tool",
                    "name": getattr(item, "tool_name", None) or getattr(raw, "name", "tool"),
                    "detail": str(getattr(raw, "arguments", "")),
                }
            )
        elif kind == "tool_call_output_item":
            events.append({"type": "tool_output", "name": "tool", "detail": _stringify(item.output)[:800]})
        elif kind == "handoff_call_item":
            raw = item.raw_item
            events.append({"type": "handoff", "name": getattr(raw, "name", "handoff"), "detail": "requested"})
        elif kind == "message_output_item":
            agent = getattr(item, "agent", None)
            events.append(
                {
                    "type": "message",
                    "name": getattr(agent, "name", "") if agent else "",
                    "detail": _stringify(getattr(result, "final_output", ""))[:800],
                }
            )
    if result.last_agent is not None:
        events.append({"type": "agent", "name": result.last_agent.name, "detail": "finished"})
    return events


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "model_dump"):
        import json

        return json.dumps(value.model_dump())
    return str(value)
