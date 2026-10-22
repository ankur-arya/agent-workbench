"""Export a workflow as Agents SDK Python the way Agent Builder offered a code download."""

from __future__ import annotations

from workbench.document import WorkflowDocument


def export_python(document: WorkflowDocument) -> str:
    agents = [node for node in document.nodes if node.type == "agent"]
    lines = [
        '"""Exported by agent-workbench. Logic nodes are comments; agent nodes are SDK objects."""',
        "",
        "import asyncio",
        "",
        "from agents import Agent, Runner, function_tool",
        "",
        "",
    ]
    tool_names: list[str] = []
    for node in agents:
        for name in node.config.get("tools") or []:
            if name not in tool_names:
                tool_names.append(str(name))
    if tool_names:
        lines.append("# Built-in tools are reproduced as stubs. Prefer workbench.tools in this repo.")
        for name in tool_names:
            lines.append("@function_tool")
            lines.append(f"def {name}(query: str) -> str:")
            lines.append(f'    """{name}."""')
            lines.append(f'    return "Implement {name}"')
            lines.append("")
            lines.append("")
    # Specialists first so handoffs can reference them.
    ordered = _agent_order(document)
    for node in ordered:
        var = _var(node.id)
        config = node.config
        tools = ", ".join(str(name) for name in config.get("tools") or [])
        handoffs = ", ".join(_var(edge.target) for edge in document.outgoing(node.id, "handoff"))
        as_tools = [edge.target for edge in document.outgoing(node.id, "tool") if document.node(edge.target).type == "agent"]
        tool_expr = f"[{tools}]" if tools else "[]"
        if as_tools:
            calls = ", ".join(
                f'{_var(target)}.as_tool(tool_name="{document.node(target).config.get("tool_name") or _var(target)}", '
                f'tool_description="{_one_line(str(document.node(target).config.get("tool_description") or "Specialist"))}")'
                for target in as_tools
            )
            tool_expr = f"{tool_expr} + [{calls}]" if tools else f"[{calls}]"
        lines.append(f"{var} = Agent(")
        lines.append(f'    name="{config.get("name") or node.id}",')
        lines.append(f'    instructions="{_one_line(str(config.get("instructions") or ""))}",')
        if config.get("handoff_description"):
            lines.append(f'    handoff_description="{_one_line(str(config["handoff_description"]))}",')
        lines.append(f"    tools={tool_expr},")
        if handoffs:
            lines.append(f"    handoffs=[{handoffs}],")
        lines.append(")")
        lines.append("")
    logic = [node for node in document.nodes if node.type not in {"start", "agent"}]
    if logic:
        lines.append("# Canvas logic nodes are executed by workbench.runtime, not by a single Runner call:")
        for node in logic:
            lines.append(f"# - {node.type} {node.id}: {node.config}")
        lines.append("")
    entry = _entry_agent(document)
    entry_var = _var(entry.id) if entry is not None else "agent"
    lines.extend(
        [
            "async def main() -> None:",
            f'    result = await Runner.run({entry_var}, "Hello")',
            "    print(result.final_output)",
            "",
            "",
            'if __name__ == "__main__":',
            "    asyncio.run(main())",
            "",
        ]
    )
    return "\n".join(lines)


def _entry_agent(document: WorkflowDocument):
    seen = {document.start().id}
    node = document.successor(document.start().id, "out") or document.successor(document.start().id, "pass")
    while node is not None and node.id not in seen:
        seen.add(node.id)
        if node.type == "agent":
            return node
        handle = "pass" if document.outgoing(node.id, "pass") else "out"
        node = document.successor(node.id, handle)
    for node in document.nodes:
        if node.type == "agent":
            return node
    return None


def _agent_order(document: WorkflowDocument) -> list:
    agents = [node for node in document.nodes if node.type == "agent"]
    dependents: dict[str, set[str]] = {node.id: set() for node in agents}
    for node in agents:
        for edge in document.outgoing(node.id, "handoff") + document.outgoing(node.id, "tool"):
            if edge.target in dependents:
                dependents[node.id].add(edge.target)
    ordered = []
    remaining = {node.id: node for node in agents}
    while remaining:
        ready = [node_id for node_id, deps in dependents.items() if node_id in remaining and not (deps & remaining.keys())]
        if not ready:
            ordered.extend(remaining.values())
            break
        for node_id in ready:
            ordered.append(remaining.pop(node_id))
    return ordered


def _var(node_id: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in node_id)
    return cleaned or "agent"


def _one_line(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
