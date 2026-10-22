"""Workflow documents: the canvas, the harness, and optional eval tasks."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

NodeType = Literal[
    "start",
    "agent",
    "note",
    "file_search",
    "guardrails",
    "mcp",
    "if_else",
    "while",
    "human_approval",
    "transform",
    "set_state",
]
ProviderName = Literal["local", "openai"]
SessionBackend = Literal["none", "memory", "sqlite"]
ApprovalMode = Literal["auto", "manual", "reject"]
ToolChoice = Literal["auto", "required", "none"]
ToolUseBehavior = Literal["run_llm_again", "stop_on_first_tool"]


class HarnessSpec(BaseModel):
    """How an agent run is executed. New harnesses register against `kind`."""

    kind: str = "sync"
    max_turns: int = 10
    tracing: bool = False
    provider: ProviderName = "local"
    model: str | None = None
    temperature: float | None = 0.2
    tool_choice: ToolChoice | None = "auto"
    parallel_tool_calls: bool | None = False
    tool_use_behavior: ToolUseBehavior = "run_llm_again"
    session_backend: SessionBackend = "none"
    session_id: str | None = None
    approval: ApprovalMode = "manual"
    eval_mode: Literal["sync", "async"] = "sync"


class EvalTask(BaseModel):
    input: str
    expect_contains: str | None = None
    expect_exact: str | None = None
    expect_tool: str | None = None
    expect_handoff: str | None = None
    expect_guardrail: bool | None = None
    expect_status: str | None = None
    similarity_reference: str | None = None
    similarity_threshold: float = 0.34
    max_words: int | None = None


class Node(BaseModel):
    id: str
    type: NodeType
    x: float = 40
    y: float = 80
    config: dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    id: str
    source: str
    target: str
    source_handle: str = "out"
    target_handle: str = "in"


class WorkflowDocument(BaseModel):
    name: str
    description: str = ""
    template: str | None = None
    harness: HarnessSpec = Field(default_factory=HarnessSpec)
    nodes: list[Node]
    edges: list[Edge] = Field(default_factory=list)
    tasks: list[EvalTask] = Field(default_factory=list)

    def node(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"Unknown node {node_id}")

    def start(self) -> Node:
        starts = [node for node in self.nodes if node.type == "start"]
        if len(starts) != 1:
            raise ValueError("A workflow needs exactly one start node")
        return starts[0]

    def outgoing(self, node_id: str, handle: str | None = None) -> list[Edge]:
        edges = [edge for edge in self.edges if edge.source == node_id]
        if handle is None:
            return edges
        return [edge for edge in edges if edge.source_handle == handle]

    def successor(self, node_id: str, handle: str = "out") -> Node | None:
        edges = self.outgoing(node_id, handle)
        if not edges:
            return None
        return self.node(edges[0].target)
