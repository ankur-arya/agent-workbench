from workbench.document import Edge, Node, WorkflowDocument
from workbench.expressions import evaluate
from workbench.runtime import run_workflow


def _graph(nodes, edges) -> WorkflowDocument:
    return WorkflowDocument(name="logic", nodes=nodes, edges=edges)


def test_expressions() -> None:
    state = {"input_as_text": "refund policy", "output": "facts", "vars": {"topic": "math"}}
    assert evaluate('contains(output, "facts")', state)
    assert evaluate('not_contains(input_as_text, "jailbreak")', state)
    assert evaluate('equals(topic, "math")', state)
    assert not evaluate("empty(output)", state)
    assert evaluate("word_count(output) < 3", state)


def test_if_else_and_transform() -> None:
    document = _graph(
        [
            Node(id="start", type="start"),
            Node(id="seed", type="transform", config={"mode": "set", "value": "facts"}),
            Node(id="branch", type="if_else", config={"expression": 'contains(output, "facts")'}),
            Node(id="yes", type="transform", config={"mode": "set", "value": "YES"}),
            Node(id="no", type="transform", config={"mode": "set", "value": "NO"}),
        ],
        [
            Edge(id="e1", source="start", target="seed"),
            Edge(id="e2", source="seed", target="branch"),
            Edge(id="e3", source="branch", target="yes", source_handle="true"),
            Edge(id="e4", source="branch", target="no", source_handle="false"),
        ],
    )
    outcome = run_workflow(document, "hello")
    assert outcome.status == "completed"
    assert outcome.output == "YES"


def test_while_appends_until_the_expression_fails() -> None:
    document = _graph(
        [
            Node(id="start", type="start"),
            Node(id="loop", type="while", config={"expression": 'not_contains(output, "xx")', "max_iterations": 4}),
            Node(id="body", type="transform", config={"mode": "append", "value": "x"}),
        ],
        [
            Edge(id="e1", source="start", target="loop"),
            Edge(id="e2", source="loop", target="body", source_handle="body"),
            Edge(id="e3", source="body", target="loop"),
        ],
    )
    outcome = run_workflow(document, "go")
    assert outcome.output == "xx"


def test_guardrail_node_blocks_email_and_mask_redacts() -> None:
    blocked = _graph(
        [
            Node(id="start", type="start"),
            Node(id="guard", type="guardrails", config={"checks": ["pii"], "action": "flag", "source": "input"}),
            Node(id="after", type="transform", config={"mode": "set", "value": "should not run"}),
        ],
        [
            Edge(id="e1", source="start", target="guard"),
            Edge(id="e2", source="guard", target="after", source_handle="pass"),
        ],
    )
    outcome = run_workflow(blocked, "mail me at ada@example.com")
    assert outcome.status == "blocked"
    assert "email" in outcome.output

    masked = _graph(
        [
            Node(id="start", type="start"),
            Node(id="guard", type="guardrails", config={"checks": ["pii"], "action": "mask", "source": "input"}),
            Node(
                id="agent",
                type="agent",
                config={"name": "Echo", "instructions": "Rewrite the user question as one clear sentence."},
            ),
        ],
        [
            Edge(id="e1", source="start", target="guard"),
            Edge(id="e2", source="guard", target="agent", source_handle="pass"),
        ],
    )
    outcome = run_workflow(masked, "mail ada@example.com about refunds")
    assert outcome.status == "completed"
    assert "ada@example.com" not in outcome.output
    assert "[email]" in outcome.output
