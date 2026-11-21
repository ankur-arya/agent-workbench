"""SQLite store for workflows, versions, threads, traces, connectors, and evals."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workbench.document import EvalTask, WorkflowDocument
from workbench.ids import new_id
from workbench.paths import workbench_home
from workbench.tools import LEGACY_CONNECTORS, TOOL_CATALOG


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Store:
    def __init__(self, home: Path | None = None) -> None:
        self.home = Path(home) if home is not None else workbench_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.path = self.home / "workbench.db"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def close(self) -> None:
        self._conn.close()

    def _init(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    template TEXT,
                    status TEXT NOT NULL,
                    document_json TEXT NOT NULL,
                    current_version INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    document_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    version_id TEXT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    trace_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    thread_id TEXT,
                    version_id TEXT,
                    harness TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input TEXT NOT NULL,
                    output TEXT NOT NULL,
                    last_agent TEXT,
                    events_json TEXT NOT NULL,
                    pending_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS connectors (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    server_url TEXT,
                    requires_approval INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    workflow_id TEXT,
                    tasks_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT,
                    workflow_id TEXT NOT NULL,
                    version_id TEXT,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()
        self._seed_connectors()

    def _seed_connectors(self) -> None:
        existing = {row["id"] for row in self.list_connectors()}
        for item in TOOL_CATALOG:
            if item["id"] in existing:
                continue
            self.upsert_connector(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "kind": "builtin",
                    "description": item["description"],
                    "enabled": True,
                    "requires_approval": item["requires_approval"],
                }
            )
        for item in LEGACY_CONNECTORS:
            if item["id"] in existing:
                continue
            self.upsert_connector(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "kind": "legacy",
                    "description": "OpenAI legacy connector. Live calls need a token and the OpenAI provider.",
                    "enabled": False,
                    "requires_approval": True,
                }
            )

    def list_connectors(self) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM connectors ORDER BY kind, name")
        return [_connector(row) for row in rows]

    def upsert_connector(self, payload: dict[str, Any]) -> dict[str, Any]:
        connector_id = str(payload.get("id") or new_id("con"))
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO connectors (id, name, kind, description, enabled, server_url, requires_approval, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    kind=excluded.kind,
                    description=excluded.description,
                    enabled=excluded.enabled,
                    server_url=excluded.server_url,
                    requires_approval=excluded.requires_approval
                """,
                (
                    connector_id,
                    payload.get("name") or connector_id,
                    payload.get("kind") or "mcp",
                    payload.get("description") or "",
                    1 if payload.get("enabled", True) else 0,
                    payload.get("server_url"),
                    1 if payload.get("requires_approval") else 0,
                    _now(),
                ),
            )
            self._conn.commit()
        return self.get_connector(connector_id)

    def get_connector(self, connector_id: str) -> dict[str, Any]:
        row = self._one("SELECT * FROM connectors WHERE id = ?", (connector_id,))
        if row is None:
            raise KeyError(f"Unknown connector {connector_id}")
        return _connector(row)

    def delete_connector(self, connector_id: str) -> None:
        current = self.get_connector(connector_id)
        if current["kind"] == "builtin":
            raise ValueError("Built-in connectors can be disabled, not deleted.")
        with self._lock:
            self._conn.execute("DELETE FROM connectors WHERE id = ?", (connector_id,))
            self._conn.commit()

    def create_workflow(self, document: WorkflowDocument) -> dict[str, Any]:
        workflow_id = new_id("wf")
        stamp = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO workflows
                (id, name, description, template, status, document_json, current_version, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'draft', ?, 0, ?, ?)
                """,
                (
                    workflow_id,
                    document.name,
                    document.description,
                    document.template,
                    document.model_dump_json(),
                    stamp,
                    stamp,
                ),
            )
            self._conn.commit()
        return self.get_workflow(workflow_id)

    def list_workflows(self) -> list[dict[str, Any]]:
        return [_workflow(row) for row in self._query("SELECT * FROM workflows ORDER BY updated_at DESC")]

    def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        row = self._one("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
        if row is None:
            raise KeyError(f"Unknown workflow {workflow_id}")
        return _workflow(row)

    def save_workflow(self, workflow_id: str, document: WorkflowDocument) -> dict[str, Any]:
        self.get_workflow(workflow_id)
        with self._lock:
            self._conn.execute(
                """
                UPDATE workflows
                SET name = ?, description = ?, template = ?, document_json = ?, status = 'draft', updated_at = ?
                WHERE id = ?
                """,
                (
                    document.name,
                    document.description,
                    document.template,
                    document.model_dump_json(),
                    _now(),
                    workflow_id,
                ),
            )
            self._conn.commit()
        return self.get_workflow(workflow_id)

    def publish(self, workflow_id: str, note: str = "") -> dict[str, Any]:
        workflow = self.get_workflow(workflow_id)
        version = int(workflow["current_version"]) + 1
        version_id = new_id("ver")
        stamp = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO versions (id, workflow_id, version, note, document_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (version_id, workflow_id, version, note, json.dumps(workflow["document"]), stamp),
            )
            self._conn.execute(
                "UPDATE workflows SET status = 'published', current_version = ?, updated_at = ? WHERE id = ?",
                (version, stamp, workflow_id),
            )
            self._conn.commit()
        return self.get_version(version_id)

    def list_versions(self, workflow_id: str) -> list[dict[str, Any]]:
        self.get_workflow(workflow_id)
        rows = self._query(
            "SELECT * FROM versions WHERE workflow_id = ? ORDER BY version DESC",
            (workflow_id,),
        )
        return [_version(row) for row in rows]

    def get_version(self, version_id: str) -> dict[str, Any]:
        row = self._one("SELECT * FROM versions WHERE id = ?", (version_id,))
        if row is None:
            raise KeyError(f"Unknown version {version_id}")
        return _version(row)

    def document_for(self, workflow_id: str, version_id: str | None = None) -> WorkflowDocument:
        if version_id:
            version = self.get_version(version_id)
            if version["workflow_id"] != workflow_id:
                raise KeyError("Version does not belong to this workflow")
            return WorkflowDocument.model_validate(version["document"])
        return WorkflowDocument.model_validate(self.get_workflow(workflow_id)["document"])

    def restore(self, workflow_id: str, version_id: str) -> dict[str, Any]:
        document = self.document_for(workflow_id, version_id)
        return self.save_workflow(workflow_id, document)

    def create_thread(self, workflow_id: str, version_id: str | None = None, title: str = "Preview") -> dict[str, Any]:
        self.get_workflow(workflow_id)
        thread_id = new_id("thr")
        with self._lock:
            self._conn.execute(
                "INSERT INTO threads (id, workflow_id, version_id, title, created_at) VALUES (?, ?, ?, ?, ?)",
                (thread_id, workflow_id, version_id, title, _now()),
            )
            self._conn.commit()
        return self.get_thread(thread_id)

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        row = self._one("SELECT * FROM threads WHERE id = ?", (thread_id,))
        if row is None:
            raise KeyError(f"Unknown thread {thread_id}")
        payload = dict(row)
        payload["messages"] = [dict(item) for item in self._query(
            "SELECT * FROM messages WHERE thread_id = ? ORDER BY created_at, id",
            (thread_id,),
        )]
        return payload

    def add_message(self, thread_id: str, role: str, content: str, trace_id: str | None = None) -> dict[str, Any]:
        message_id = new_id("msg")
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (id, thread_id, role, content, trace_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, thread_id, role, content, trace_id, _now()),
            )
            self._conn.commit()
        row = self._one("SELECT * FROM messages WHERE id = ?", (message_id,))
        return dict(row) if row else {}

    def add_trace(self, payload: dict[str, Any]) -> dict[str, Any]:
        trace_id = payload.get("id") or new_id("tr")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO traces
                (id, workflow_id, thread_id, version_id, harness, status, input, output, last_agent, events_json, pending_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    payload["workflow_id"],
                    payload.get("thread_id"),
                    payload.get("version_id"),
                    payload.get("harness") or "sync",
                    payload.get("status") or "completed",
                    payload.get("input") or "",
                    payload.get("output") or "",
                    payload.get("last_agent"),
                    json.dumps(payload.get("events") or []),
                    json.dumps(payload["pending"]) if payload.get("pending") else None,
                    _now(),
                ),
            )
            self._conn.commit()
        return self.get_trace(str(trace_id))

    def update_trace(self, trace_id: str, **changes: Any) -> dict[str, Any]:
        current = self.get_trace(trace_id)
        events = changes.get("events", current["events"])
        pending = changes.get("pending", current["pending"])
        with self._lock:
            self._conn.execute(
                """
                UPDATE traces
                SET status = ?, output = ?, last_agent = ?, events_json = ?, pending_json = ?
                WHERE id = ?
                """,
                (
                    changes.get("status", current["status"]),
                    changes.get("output", current["output"]),
                    changes.get("last_agent", current["last_agent"]),
                    json.dumps(events),
                    json.dumps(pending) if pending else None,
                    trace_id,
                ),
            )
            self._conn.commit()
        return self.get_trace(trace_id)

    def list_traces(self, workflow_id: str) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM traces WHERE workflow_id = ? ORDER BY created_at DESC",
            (workflow_id,),
        )
        return [_trace(row) for row in rows]

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        row = self._one("SELECT * FROM traces WHERE id = ?", (trace_id,))
        if row is None:
            raise KeyError(f"Unknown trace {trace_id}")
        return _trace(row)

    def create_dataset(self, name: str, tasks: list[dict[str, Any]], workflow_id: str | None = None) -> dict[str, Any]:
        dataset_id = new_id("ds")
        with self._lock:
            self._conn.execute(
                "INSERT INTO datasets (id, name, workflow_id, tasks_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (dataset_id, name, workflow_id, json.dumps(tasks), _now()),
            )
            self._conn.commit()
        return self.get_dataset(dataset_id)

    def list_datasets(self) -> list[dict[str, Any]]:
        return [_dataset(row) for row in self._query("SELECT * FROM datasets ORDER BY created_at DESC")]

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        row = self._one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        if row is None:
            raise KeyError(f"Unknown dataset {dataset_id}")
        return _dataset(row)

    def add_eval_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        eval_id = new_id("ev")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO eval_runs (id, dataset_id, workflow_id, version_id, summary_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    eval_id,
                    payload.get("dataset_id"),
                    payload["workflow_id"],
                    payload.get("version_id"),
                    json.dumps(payload.get("summary") or {}),
                    _now(),
                ),
            )
            self._conn.commit()
        row = self._one("SELECT * FROM eval_runs WHERE id = ?", (eval_id,))
        return _eval_run(row) if row else {}

    def list_eval_runs(self, workflow_id: str | None = None) -> list[dict[str, Any]]:
        if workflow_id:
            rows = self._query("SELECT * FROM eval_runs WHERE workflow_id = ? ORDER BY created_at DESC", (workflow_id,))
        else:
            rows = self._query("SELECT * FROM eval_runs ORDER BY created_at DESC")
        return [_eval_run(row) for row in rows]

    def stats(self) -> dict[str, int]:
        def count(table: str) -> int:
            row = self._one(f"SELECT COUNT(*) AS n FROM {table}")
            return int(row["n"]) if row else 0

        return {
            "workflows": count("workflows"),
            "versions": count("versions"),
            "traces": count("traces"),
            "connectors": count("connectors"),
            "datasets": count("datasets"),
        }

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params))

    def _one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        rows = self._query(sql, params)
        return rows[0] if rows else None


def tasks_from_document(document: WorkflowDocument) -> list[dict[str, Any]]:
    return [EvalTask.model_validate(task).model_dump(mode="json") for task in document.tasks]


def _loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    return json.loads(value)


def _workflow(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["document"] = json.loads(payload.pop("document_json"))
    return payload


def _version(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["document"] = json.loads(payload.pop("document_json"))
    return payload


def _trace(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["events"] = _loads(payload.pop("events_json"), [])
    payload["pending"] = _loads(payload.pop("pending_json"), None)
    return payload


def _connector(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["enabled"] = bool(payload["enabled"])
    payload["requires_approval"] = bool(payload["requires_approval"])
    return payload


def _dataset(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["tasks"] = _loads(payload.pop("tasks_json"), [])
    return payload


def _eval_run(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["summary"] = _loads(payload.pop("summary_json"), {})
    return payload
