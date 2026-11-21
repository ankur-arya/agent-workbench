"""Descriptions the UI and CLI show for nodes and harnesses."""

from __future__ import annotations

from workbench.harness import HARNESSES

NODE_TYPES = [
    {"type": "start", "group": "Core", "summary": "Entry point. Exposes input_as_text from the user."},
    {"type": "agent", "group": "Core", "summary": "Instructions, model, tools, handoffs, guardrails, and output type."},
    {"type": "note", "group": "Core", "summary": "Comment on the canvas. Notes do not run."},
    {"type": "file_search", "group": "Tools", "summary": "Search a local corpus, or attach file search to an agent."},
    {"type": "guardrails", "group": "Tools", "summary": "Flag or mask PII, jailbreaks, and blocked keywords."},
    {"type": "mcp", "group": "Tools", "summary": "Attach a connector from the registry."},
    {"type": "if_else", "group": "Logic", "summary": "Branch with contains, equals, empty, or word_count."},
    {"type": "while", "group": "Logic", "summary": "Repeat the body edge while an expression is true."},
    {"type": "human_approval", "group": "Logic", "summary": "Pause until a person approves or rejects the step."},
    {"type": "transform", "group": "Data", "summary": "Wrap, replace, append, or extract JSON from the output."},
    {"type": "set_state", "group": "Data", "summary": "Store input or output in a named variable."},
]

HARNESS_INFO = [
    {
        "kind": "sync",
        "summary": "Blocking run via the SDK runner. Returns one final result.",
    },
    {
        "kind": "async",
        "summary": "Same loop as sync, awaited on the event loop.",
    },
    {
        "kind": "streaming",
        "summary": "Runner.run_streamed. Token deltas are returned with the trace.",
    },
    {
        "kind": "approval",
        "summary": "Pause, auto-approve, or reject when a tool needs human approval.",
    },
    {
        "kind": "eval",
        "summary": "Run the workflow against a dataset and grade each trace.",
    },
]


def harness_catalog() -> list[dict[str, str]]:
    known = {item["kind"] for item in HARNESS_INFO}
    extra = [{"kind": name, "summary": "Registered harness."} for name in HARNESSES if name not in known]
    return HARNESS_INFO + extra
