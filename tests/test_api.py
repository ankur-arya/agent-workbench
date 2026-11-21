from fastapi.testclient import TestClient

from workbench.api import create_app
from workbench.store import Store


def test_create_run_publish_and_preview(tmp_path) -> None:
    client = TestClient(create_app(Store(tmp_path)))
    created = client.post("/api/workflows", json={"template": "tool_assistant", "name": "Desk"}).json()
    workflow_id = created["id"]

    run = client.post(f"/api/workflows/{workflow_id}/run", json={"input": "What is 2 + 2?"})
    assert run.status_code == 200
    body = run.json()
    assert "4" in body["outcome"]["output"]
    assert body["trace"]["status"] == "completed"

    published = client.post(f"/api/workflows/{workflow_id}/publish", json={"note": "first"}).json()
    assert published["version"] == 1

    edited = created["document"]
    edited["nodes"][1]["config"]["instructions"] = "Reply with the word edited."
    saved = client.put(f"/api/workflows/{workflow_id}", json={"document": edited})
    assert saved.status_code == 200
    assert saved.json()["status"] == "draft"

    restored = client.post(f"/api/workflows/{workflow_id}/restore", json={"version_id": published["id"]})
    assert "calculator" in restored.json()["document"]["nodes"][1]["config"]["tools"]

    thread = client.post("/api/threads", json={"workflow_id": workflow_id, "title": "Preview"}).json()
    message = client.post(f"/api/threads/{thread['id']}/messages", json={"text": "What is 2 + 2?"})
    assert message.status_code == 200
    assert "4" in message.json()["outcome"]["output"]

    graded = client.post(
        f"/api/traces/{body['trace']['id']}/grade",
        json={"expect_contains": "4", "expect_tool": "calculator"},
    )
    assert graded.json()["passed"] is True

    code = client.get(f"/api/workflows/{workflow_id}/code").json()["python"]
    assert "Agent(" in code and "Runner.run" in code


def test_eval_and_connectors_and_streaming(tmp_path) -> None:
    client = TestClient(create_app(Store(tmp_path)))
    created = client.post("/api/workflows", json={"template": "guardrailed"}).json()
    workflow_id = created["id"]
    report = client.post(f"/api/workflows/{workflow_id}/eval", json={})
    assert report.status_code == 200
    summary = report.json()["outcome"]["report"]
    assert summary["failed"] == 0
    assert summary["passed"] == 2

    connectors = client.get("/api/connectors").json()
    assert any(item["id"] == "connector_dropbox" and item["enabled"] is False for item in connectors)
    added = client.post(
        "/api/connectors",
        json={"name": "Local notes", "kind": "mcp", "server_url": "http://127.0.0.1:9/mcp", "enabled": True},
    )
    assert added.status_code == 200

    document = created["document"]
    document["harness"]["kind"] = "streaming"
    client.put(f"/api/workflows/{workflow_id}", json={"document": document})
    thread = client.post("/api/threads", json={"workflow_id": workflow_id}).json()
    with client.stream("POST", f"/api/threads/{thread['id']}/messages", json={"text": "What is 2 + 2?", "stream": True}) as response:
        raw = "".join(response.iter_text())
    assert "data:" in raw
    assert "4" in raw


def test_approval_resume_endpoint(tmp_path) -> None:
    client = TestClient(create_app(Store(tmp_path)))
    created = client.post("/api/workflows", json={"template": "review"}).json()
    run = client.post(
        f"/api/workflows/{created['id']}/run",
        json={"input": "Please draft a note that the refund was approved."},
    ).json()
    assert run["outcome"]["status"] == "needs_approval"
    resumed = client.post(f"/api/traces/{run['trace']['id']}/resume", json={"decision": "approve"})
    assert resumed.status_code == 200
    assert "Sent:" in resumed.json()["outcome"]["output"]
