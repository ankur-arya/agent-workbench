"""Built-in function tools. They run locally so demos and tests need no API key."""

from __future__ import annotations

import ast
import operator
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agents import function_tool
from agents.tool import FunctionTool

CORPUS = Path(__file__).resolve().parent / "corpus"
_NOTES: list[str] = []
_ORDERS = {
    "1001": "Order 1001 is shipped.",
    "1002": "Order 1002 is processing.",
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression using numbers and + - * / % parentheses."""
    try:
        value = _safe_calc(expression)
    except Exception as exc:
        return f"Could not evaluate {expression!r}: {exc}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def clock() -> str:
    """Return the current time in UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def take_note(text: str) -> str:
    """Save a short note and confirm it was stored."""
    _NOTES.append(text)
    return f"Saved note: {text}"


def lookup_order(order_id: str) -> str:
    """Look up an order by its numeric id."""
    return _ORDERS.get(order_id.strip(), f"No order found for {order_id}.")


def send_note(text: str) -> str:
    """Send a note to the user. Call this only when the draft is approved."""
    return f"Sent: {text}"


def file_search(query: str) -> str:
    """Search the local handbook for passages that match the query."""
    return _search_files(query, [CORPUS / "handbook.md"])


def web_search(query: str) -> str:
    """Search the local research index. This is the offline stand-in for hosted web search."""
    return _search_files(query, [CORPUS / "web.md", CORPUS / "handbook.md"])


def mcp_stub(server_label: str, query: str) -> str:
    """Describe a registered MCP connector without calling a remote server."""
    return (
        f"Connector {server_label} is registered. "
        f"No remote MCP call was made for {query!r}. "
        "Use provider openai with a server URL or connector token to attach a hosted MCP tool."
    )


RAW_TOOLS: dict[str, Callable[..., str]] = {
    "calculator": calculator,
    "clock": clock,
    "take_note": take_note,
    "lookup_order": lookup_order,
    "send_note": send_note,
    "file_search": file_search,
    "web_search": web_search,
}

TOOL_CATALOG: list[dict[str, Any]] = [
    {
        "id": "calculator",
        "name": "Calculator",
        "kind": "builtin",
        "description": "Arithmetic with + - * / and parentheses.",
        "requires_approval": False,
    },
    {
        "id": "clock",
        "name": "Clock",
        "kind": "builtin",
        "description": "Current UTC time.",
        "requires_approval": False,
    },
    {
        "id": "take_note",
        "name": "Notes",
        "kind": "builtin",
        "description": "Save a note in the local workbench.",
        "requires_approval": False,
    },
    {
        "id": "lookup_order",
        "name": "Order lookup",
        "kind": "builtin",
        "description": "Look up a sample order by id.",
        "requires_approval": False,
    },
    {
        "id": "send_note",
        "name": "Send note",
        "kind": "builtin",
        "description": "Send a drafted note. Requires approval before it runs.",
        "requires_approval": True,
    },
    {
        "id": "file_search",
        "name": "File search",
        "kind": "builtin",
        "description": "Search the bundled handbook. Stand-in for hosted file search.",
        "requires_approval": False,
    },
    {
        "id": "web_search",
        "name": "Web search",
        "kind": "builtin",
        "description": "Search the local research index. Stand-in for hosted web search.",
        "requires_approval": False,
    },
]

LEGACY_CONNECTORS: list[dict[str, str]] = [
    {"id": "connector_dropbox", "name": "Dropbox"},
    {"id": "connector_gmail", "name": "Gmail"},
    {"id": "connector_googlecalendar", "name": "Google Calendar"},
    {"id": "connector_googledrive", "name": "Google Drive"},
    {"id": "connector_microsoftteams", "name": "Microsoft Teams"},
    {"id": "connector_outlookcalendar", "name": "Outlook Calendar"},
    {"id": "connector_outlookemail", "name": "Outlook Email"},
    {"id": "connector_sharepoint", "name": "SharePoint"},
]


def make_function_tool(name: str, *, needs_approval: bool = False) -> FunctionTool:
    fn = RAW_TOOLS.get(name)
    if fn is None:
        raise KeyError(f"Unknown built-in tool {name}")
    return function_tool(fn, name_override=name, needs_approval=needs_approval)


def make_mcp_stub_tool(server_label: str) -> FunctionTool:
    label = server_label

    def connector_query(query: str) -> str:
        """Query a registered connector. Remote MCP is not called from the local provider."""
        return mcp_stub(label, query)

    safe = re.sub(r"[^a-zA-Z0-9_]", "_", label).lower() or "connector"
    return function_tool(connector_query, name_override=f"mcp_{safe}")


def _search_files(query: str, paths: list[Path]) -> str:
    terms = [part.lower() for part in re.findall(r"[A-Za-z0-9']+", query) if len(part) > 2]
    chunks: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        paragraphs = [block.strip() for block in path.read_text(encoding="utf-8").split("\n\n")]
        for paragraph in paragraphs:
            haystack = paragraph.lower()
            if any(term in haystack for term in terms):
                chunks.append(paragraph.replace("\n", " "))
    if not chunks:
        return f"No local passages matched {query!r}."
    return "\n".join(chunks[:4])


def _safe_calc(expression: str) -> float | int:
    tree = ast.parse(expression, mode="eval")

    def walk(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("only numbers and + - * / % are allowed")

    return walk(tree)
