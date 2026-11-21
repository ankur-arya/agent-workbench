"""A small expression language for if/else and while nodes.

This is the local stand-in for Agent Builder's CEL expressions. It does not
call Python `eval`. Supported forms:

- contains(field, "text")
- not_contains(field, "text")
- equals(field, "text")
- not_empty(field)
- empty(field)
- word_count(field) < N
- word_count(field) <= N
"""

from __future__ import annotations

import re
from typing import Any

_CONTAINS = re.compile(r'^(not_contains|contains)\(\s*([A-Za-z_][\w]*)\s*,\s*"([^"]*)"\s*\)$')
_EQUALS = re.compile(r'^equals\(\s*([A-Za-z_][\w]*)\s*,\s*"([^"]*)"\s*\)$')
_EMPTY = re.compile(r"^(not_empty|empty)\(\s*([A-Za-z_][\w]*)\s*\)$")
_WORDS = re.compile(r"^word_count\(\s*([A-Za-z_][\w]*)\s*\)\s*(<=|<)\s*(\d+)$")


def evaluate(expression: str, state: dict[str, Any]) -> bool:
    expr = " ".join(expression.strip().split())
    if not expr:
        raise ValueError("Expression is empty")

    match = _CONTAINS.match(expr)
    if match:
        kind, field, needle = match.groups()
        haystack = _text(state, field).lower()
        found = needle.lower() in haystack
        return found if kind == "contains" else not found

    match = _EQUALS.match(expr)
    if match:
        field, expected = match.groups()
        return _text(state, field).strip() == expected

    match = _EMPTY.match(expr)
    if match:
        kind, field = match.groups()
        blank = not _text(state, field).strip()
        return (not blank) if kind == "not_empty" else blank

    match = _WORDS.match(expr)
    if match:
        field, op, raw = match.groups()
        count = len(_text(state, field).split())
        limit = int(raw)
        return count < limit if op == "<" else count <= limit

    raise ValueError(
        "Unsupported expression. Use contains, not_contains, equals, empty, not_empty, or word_count."
    )


def _text(state: dict[str, Any], field: str) -> str:
    if field == "input_as_text":
        return str(state.get("input_as_text") or "")
    if field == "output":
        return str(state.get("output") or "")
    variables = state.get("vars") or {}
    if field in variables:
        return str(variables[field])
    if field in state:
        return str(state[field] or "")
    return ""
