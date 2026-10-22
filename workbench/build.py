"""Compile canvas nodes into OpenAI Agents SDK `Agent` objects."""

from __future__ import annotations

from typing import Any

from agents import Agent, HostedMCPTool, InputGuardrail, OutputGuardrail, input_guardrail, output_guardrail
from agents.guardrail import GuardrailFunctionOutput
from pydantic import BaseModel, create_model

from workbench.checks import findings_for
from workbench.document import HarnessSpec, Node, WorkflowDocument
from workbench.env import connector_token
from workbench.tools import RAW_TOOLS, TOOL_CATALOG, make_function_tool, make_mcp_stub_tool

_TYPE_MAP = {"string": str, "number": float, "integer": int, "boolean": bool}


def define_agent(
    *,
    name: str,
    instructions: str,
    tools: list[str] | None = None,
    handoffs: list[Agent[Any]] | None = None,
    model: str = "local",
    tool_use_behavior: str = "run_llm_again",
    input_guardrails: list[str] | None = None,
    output_guardrails: list[str] | None = None,
    output_fields: list[dict[str, str]] | None = None,
    needs_approval: bool = False,
) -> Agent[Any]:
    """Build an SDK agent from code. Tool names refer to the built-in registry."""
    tool_objs = [
        make_function_tool(tool_name, needs_approval=needs_approval and _requires_approval(tool_name))
        for tool_name in tools or []
    ]
    return Agent(
        name=name,
        instructions=instructions,
        tools=tool_objs,
        handoffs=list(handoffs or []),
        model=model,
        tool_use_behavior=tool_use_behavior,  # type: ignore[arg-type]
        input_guardrails=_input_guards(input_guardrails or [], "flag"),
        output_guardrails=_output_guards(output_guardrails or [], "flag"),
        output_type=_output_model(output_fields or []),
        handoff_description=instructions[:180],
    )


def build_agent(
    node: Node,
    document: WorkflowDocument,
    spec: HarnessSpec,
    *,
    connectors: list[dict[str, Any]] | None = None,
    stack: tuple[str, ...] = (),
) -> Agent[Any]:
    if node.id in stack:
        raise ValueError(f"Cycle while building agent {node.id}")
    config = node.config
    name = str(config.get("name") or node.id)
    instructions = str(config.get("instructions") or "You are a helpful agent.")
    behavior = config.get("tool_use_behavior") or spec.tool_use_behavior
    tool_objs: list[Any] = []
    for tool_name in _tool_names(node, document):
        tool_objs.append(_tool_for(tool_name, spec, connectors))
    for edge in document.outgoing(node.id, "tool"):
        target = document.node(edge.target)
        if target.type != "agent":
            continue
        specialist = build_agent(target, document, spec, connectors=connectors, stack=stack + (node.id,))
        tool_name = str(target.config.get("tool_name") or _snake(specialist.name))
        description = str(target.config.get("tool_description") or target.config.get("handoff_description") or specialist.instructions[:180])
        tool_objs.append(
            specialist.as_tool(
                tool_name=tool_name,
                tool_description=description,
                max_turns=spec.max_turns,
            )
        )
    handoffs = []
    for edge in document.outgoing(node.id, "handoff"):
        target = document.node(edge.target)
        if target.type != "agent":
            raise ValueError(f"Handoff edge {edge.id} must target an agent")
        specialist = build_agent(target, document, spec, connectors=connectors, stack=stack + (node.id,))
        if target.config.get("handoff_description"):
            specialist.handoff_description = str(target.config["handoff_description"])
        handoffs.append(specialist)
    return Agent(
        name=name,
        instructions=instructions,
        handoff_description=str(config.get("handoff_description") or "") or None,
        tools=tool_objs,
        handoffs=handoffs,
        model=str(config.get("model") or spec.model or "local"),
        tool_use_behavior=behavior,  # type: ignore[arg-type]
        input_guardrails=_input_guards(list(config.get("input_guardrails") or []), str(config.get("guardrail_action") or "flag")),
        output_guardrails=_output_guards(list(config.get("output_guardrails") or []), str(config.get("guardrail_action") or "flag")),
        output_type=_output_model(list(config.get("output_fields") or [])),
    )


def _tool_names(node: Node, document: WorkflowDocument) -> list[str]:
    names = [str(name) for name in node.config.get("tools") or []]
    for edge in document.outgoing(node.id, "tool"):
        target = document.node(edge.target)
        if target.type == "file_search" and "file_search" not in names:
            names.append("file_search")
        elif target.type == "mcp":
            connector_id = str(target.config.get("connector_id") or "")
            if connector_id and connector_id not in names:
                names.append(connector_id)
    return names


def _tool_for(name: str, spec: HarnessSpec, connectors: list[dict[str, Any]] | None) -> Any:
    record = _record(name, connectors)
    if record is not None and not record.get("enabled", True):
        raise RuntimeError(f"Connector {name} is disabled in the registry.")
    if name in RAW_TOOLS:
        approval = _requires_approval(name) and spec.approval != "auto"
        return make_function_tool(name, needs_approval=approval)
    if record is None:
        raise KeyError(f"Unknown tool or connector {name}")
    return _connector_tool(record, spec)


def _connector_tool(record: dict[str, Any], spec: HarnessSpec) -> Any:
    kind = record.get("kind")
    label = str(record.get("name") or record["id"])
    if spec.provider == "openai" and kind == "legacy":
        token = connector_token(str(record["id"]))
        if not token:
            raise RuntimeError(
                f"{label} needs AGENT_WORKBENCH_{str(record['id']).upper()}_TOKEN before it can run on the OpenAI provider."
            )
        return HostedMCPTool(
            tool_config={
                "type": "mcp",
                "server_label": label,
                "connector_id": record["id"],
                "authorization": token,
                "require_approval": "never" if spec.approval == "auto" else "always",
            }
        )
    if spec.provider == "openai" and kind == "mcp" and record.get("server_url"):
        config: dict[str, Any] = {
            "type": "mcp",
            "server_label": label,
            "server_url": record["server_url"],
            "require_approval": "never" if spec.approval == "auto" else "always",
        }
        token = connector_token(str(record["id"]))
        if token:
            config["authorization"] = token
        return HostedMCPTool(tool_config=config)
    return make_mcp_stub_tool(label)


def _record(name: str, connectors: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if connectors is None:
        return None
    for record in connectors:
        if record.get("id") == name:
            return record
    return None


def _requires_approval(name: str) -> bool:
    for item in TOOL_CATALOG:
        if item["id"] == name:
            return bool(item["requires_approval"])
    return False


def _input_guards(checks: list[str], action: str) -> list[InputGuardrail[Any]]:
    if not checks:
        return []

    @input_guardrail(name="workbench_input_" + "_".join(checks), run_in_parallel=False)
    def _guard(_ctx: Any, _agent: Agent[Any], incoming: str | list[Any]) -> GuardrailFunctionOutput:
        text = incoming if isinstance(incoming, str) else str(incoming)
        found = findings_for(text, checks)
        return GuardrailFunctionOutput(
            output_info={"findings": found, "action": action},
            tripwire_triggered=bool(found) and action != "mask",
        )

    return [_guard]


def _output_guards(checks: list[str], action: str) -> list[OutputGuardrail[Any]]:
    if not checks:
        return []

    @output_guardrail(name="workbench_output_" + "_".join(checks))
    def _guard(_ctx: Any, _agent: Agent[Any], agent_output: Any) -> GuardrailFunctionOutput:
        text = agent_output if isinstance(agent_output, str) else str(agent_output)
        found = findings_for(text, checks)
        return GuardrailFunctionOutput(
            output_info={"findings": found, "action": action},
            tripwire_triggered=bool(found) and action != "mask",
        )

    return [_guard]


def _output_model(fields: list[dict[str, Any]]) -> type[BaseModel] | None:
    if not fields:
        return None
    kwargs: dict[str, Any] = {}
    for field in fields:
        field_type = _TYPE_MAP.get(str(field.get("type") or "string"), str)
        kwargs[str(field["name"])] = (field_type, ...)
    return create_model("WorkflowOutput", **kwargs)


def _snake(name: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in name)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "agent"
