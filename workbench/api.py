"""REST API the workbench UI talks to."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from workbench import __version__
from workbench.catalog import NODE_TYPES, harness_catalog
from workbench.document import EvalTask, WorkflowDocument
from workbench.evals import grade_trace, run_eval
from workbench.export import export_python
from workbench.paths import workbench_home
from workbench.runtime import arun_workflow
from workbench.store import Store
from workbench.templates import get_template, list_templates
from workbench.yamlio import dump_yaml, load_document

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


class WorkflowCreate(BaseModel):
    template: str | None = None
    name: str | None = None
    document: dict[str, Any] | None = None


class WorkflowUpdate(BaseModel):
    document: dict[str, Any]


class PublishBody(BaseModel):
    note: str = ""


class RestoreBody(BaseModel):
    version_id: str


class RunBody(BaseModel):
    input: str
    version_id: str | None = None
    harness: dict[str, Any] | None = None


class ThreadCreate(BaseModel):
    workflow_id: str
    version_id: str | None = None
    title: str = "Preview"


class MessageBody(BaseModel):
    text: str
    stream: bool = False


class ResumeBody(BaseModel):
    decision: str = "approve"


class ConnectorBody(BaseModel):
    id: str | None = None
    name: str
    kind: str = "mcp"
    description: str = ""
    enabled: bool = True
    server_url: str | None = None
    requires_approval: bool = False


class ConnectorPatch(BaseModel):
    enabled: bool | None = None
    name: str | None = None
    description: str | None = None
    server_url: str | None = None


class DatasetBody(BaseModel):
    name: str
    workflow_id: str | None = None
    tasks: list[dict[str, Any]] = Field(default_factory=list)


class EvalBody(BaseModel):
    workflow_id: str | None = None
    version_id: str | None = None
    dataset_id: str | None = None


class GradeBody(BaseModel):
    input: str = ""
    expect_contains: str | None = None
    expect_exact: str | None = None
    expect_tool: str | None = None
    expect_handoff: str | None = None
    expect_guardrail: bool | None = None
    expect_status: str | None = None
    similarity_reference: str | None = None
    similarity_threshold: float = 0.34
    max_words: int | None = None


class ImportBody(BaseModel):
    yaml: str
    name: str | None = None


def create_app(store: Store | None = None) -> FastAPI:
    app = FastAPI(title="Agent Workbench", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def db() -> Store:
        return store or _default_store()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "home": str(workbench_home())}

    @app.get("/api/bootstrap")
    def bootstrap() -> dict[str, Any]:
        database = db()
        return {
            "version": __version__,
            "home": str(database.home),
            "stats": database.stats(),
            "templates": list_templates(),
            "harnesses": harness_catalog(),
            "nodes": NODE_TYPES,
            "tools": database.list_connectors(),
        }

    @app.get("/api/templates")
    def templates() -> list[dict[str, str]]:
        return list_templates()

    @app.get("/api/templates/{template_id}")
    def template_detail(template_id: str) -> dict[str, Any]:
        try:
            document = get_template(template_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return document.model_dump(mode="json")

    @app.get("/api/workflows")
    def workflows() -> list[dict[str, Any]]:
        return db().list_workflows()

    @app.post("/api/workflows")
    def create_workflow(body: WorkflowCreate) -> dict[str, Any]:
        try:
            if body.document:
                document = WorkflowDocument.model_validate(body.document)
                if body.name:
                    document.name = body.name
            elif body.template:
                document = get_template(body.template, body.name)
            else:
                document = get_template("blank", body.name)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return db().create_workflow(document)

    @app.post("/api/workflows/import")
    def import_workflow(body: ImportBody) -> dict[str, Any]:
        try:
            document = load_document_text(body.yaml)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        if body.name:
            document.name = body.name
        return db().create_workflow(document)

    @app.get("/api/workflows/{workflow_id}")
    def get_workflow(workflow_id: str) -> dict[str, Any]:
        return _or_404(lambda: db().get_workflow(workflow_id))

    @app.put("/api/workflows/{workflow_id}")
    def save_workflow(workflow_id: str, body: WorkflowUpdate) -> dict[str, Any]:
        try:
            document = WorkflowDocument.model_validate(body.document)
            return db().save_workflow(workflow_id, document)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/workflows/{workflow_id}/yaml")
    def workflow_yaml(workflow_id: str) -> dict[str, str]:
        workflow = _or_404(lambda: db().get_workflow(workflow_id))
        document = WorkflowDocument.model_validate(workflow["document"])
        return {"yaml": dump_yaml(document)}

    @app.get("/api/workflows/{workflow_id}/code")
    def workflow_code(workflow_id: str) -> dict[str, str]:
        workflow = _or_404(lambda: db().get_workflow(workflow_id))
        document = WorkflowDocument.model_validate(workflow["document"])
        return {"python": export_python(document)}

    @app.post("/api/workflows/{workflow_id}/publish")
    def publish(workflow_id: str, body: PublishBody) -> dict[str, Any]:
        return _or_404(lambda: db().publish(workflow_id, body.note))

    @app.get("/api/workflows/{workflow_id}/versions")
    def versions(workflow_id: str) -> list[dict[str, Any]]:
        return _or_404(lambda: db().list_versions(workflow_id))

    @app.post("/api/workflows/{workflow_id}/restore")
    def restore(workflow_id: str, body: RestoreBody) -> dict[str, Any]:
        try:
            return db().restore(workflow_id, body.version_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/workflows/{workflow_id}/run")
    async def run(workflow_id: str, body: RunBody) -> dict[str, Any]:
        database = db()
        try:
            document = database.document_for(workflow_id, body.version_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        if body.harness:
            document.harness = document.harness.model_copy(update=body.harness)
        outcome = await arun_workflow(
            document,
            body.input,
            connectors=database.list_connectors(),
            workflow_name=document.name,
        )
        trace = database.add_trace(
            {
                "workflow_id": workflow_id,
                "version_id": body.version_id,
                "harness": outcome.harness,
                "status": outcome.status,
                "input": body.input,
                "output": outcome.output,
                "last_agent": outcome.last_agent,
                "events": outcome.events,
                "pending": outcome.pending,
            }
        )
        return {"trace": trace, "outcome": outcome.as_dict()}

    @app.post("/api/workflows/{workflow_id}/eval")
    async def eval_workflow(workflow_id: str, body: EvalBody | None = None) -> dict[str, Any]:
        database = db()
        version_id = body.version_id if body else None
        try:
            document = database.document_for(workflow_id, version_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        tasks = None
        dataset_id = body.dataset_id if body else None
        if dataset_id:
            dataset = _or_404(lambda: database.get_dataset(dataset_id))
            tasks = [EvalTask.model_validate(task) for task in dataset["tasks"]]
        elif document.tasks:
            tasks = document.tasks
        else:
            raise HTTPException(400, "Add tasks to the workflow or choose a dataset.")
        outcome = await run_eval(document, tasks, connectors=database.list_connectors())
        saved = database.add_eval_run(
            {
                "dataset_id": dataset_id,
                "workflow_id": workflow_id,
                "version_id": version_id,
                "summary": outcome.report,
            }
        )
        return {"eval": saved, "outcome": outcome.as_dict()}

    @app.get("/api/workflows/{workflow_id}/traces")
    def traces(workflow_id: str) -> list[dict[str, Any]]:
        return _or_404(lambda: db().list_traces(workflow_id))

    @app.get("/api/traces/{trace_id}")
    def trace(trace_id: str) -> dict[str, Any]:
        return _or_404(lambda: db().get_trace(trace_id))

    @app.post("/api/traces/{trace_id}/grade")
    def grade(trace_id: str, body: GradeBody) -> dict[str, Any]:
        trace_row = _or_404(lambda: db().get_trace(trace_id))
        task = EvalTask.model_validate(body.model_dump())
        if not task.input:
            task.input = trace_row["input"]
        return grade_trace(trace_row, task)

    @app.post("/api/traces/{trace_id}/resume")
    async def resume(trace_id: str, body: ResumeBody) -> dict[str, Any]:
        database = db()
        trace_row = _or_404(lambda: database.get_trace(trace_id))
        pending = trace_row.get("pending")
        if not pending:
            raise HTTPException(400, "This trace is not waiting for approval.")
        if body.decision not in {"approve", "reject"}:
            raise HTTPException(400, "Decision must be approve or reject.")
        document = database.document_for(trace_row["workflow_id"], trace_row.get("version_id"))
        spec = document.harness.model_copy(deep=True)
        if trace_row.get("thread_id"):
            spec.session_backend = "sqlite"
            spec.session_id = trace_row["thread_id"]
        outcome = await arun_workflow(
            document,
            trace_row["input"],
            harness=spec,
            connectors=database.list_connectors(),
            resume={**pending, "decision": body.decision},
            workflow_name=document.name,
        )
        updated = database.update_trace(
            trace_id,
            status=outcome.status,
            output=outcome.output,
            last_agent=outcome.last_agent,
            events=outcome.events,
            pending=outcome.pending,
        )
        if trace_row.get("thread_id"):
            database.add_message(trace_row["thread_id"], "assistant", outcome.output, trace_id)
        return {"trace": updated, "outcome": outcome.as_dict()}

    @app.post("/api/threads")
    def create_thread(body: ThreadCreate) -> dict[str, Any]:
        try:
            return db().create_thread(body.workflow_id, body.version_id, body.title)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/threads/{thread_id}")
    def get_thread(thread_id: str) -> dict[str, Any]:
        return _or_404(lambda: db().get_thread(thread_id))

    @app.post("/api/threads/{thread_id}/messages")
    async def post_message(thread_id: str, body: MessageBody) -> Any:
        database = db()
        thread = _or_404(lambda: database.get_thread(thread_id))
        document = database.document_for(thread["workflow_id"], thread.get("version_id"))
        spec = document.harness.model_copy(deep=True)
        if spec.session_backend == "none":
            spec.session_backend = "sqlite"
        spec.session_id = thread_id
        database.add_message(thread_id, "user", body.text)

        async def _go() -> dict[str, Any]:
            outcome = await arun_workflow(
                document,
                body.text,
                harness=spec,
                connectors=database.list_connectors(),
                workflow_name=document.name,
            )
            trace = database.add_trace(
                {
                    "workflow_id": thread["workflow_id"],
                    "thread_id": thread_id,
                    "version_id": thread.get("version_id"),
                    "harness": outcome.harness,
                    "status": outcome.status,
                    "input": body.text,
                    "output": outcome.output,
                    "last_agent": outcome.last_agent,
                    "events": outcome.events,
                    "pending": outcome.pending,
                }
            )
            database.add_message(thread_id, "assistant", outcome.output, trace["id"])
            return {"trace": trace, "outcome": outcome.as_dict(), "thread": database.get_thread(thread_id)}

        if body.stream or spec.kind == "streaming":

            async def events():
                queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

                async def on_delta(text: str) -> None:
                    await queue.put({"type": "delta", "text": text})

                async def runner() -> None:
                    try:
                        outcome = await arun_workflow(
                            document,
                            body.text,
                            harness=spec,
                            connectors=database.list_connectors(),
                            workflow_name=document.name,
                            on_delta=on_delta,
                        )
                        trace = database.add_trace(
                            {
                                "workflow_id": thread["workflow_id"],
                                "thread_id": thread_id,
                                "version_id": thread.get("version_id"),
                                "harness": outcome.harness,
                                "status": outcome.status,
                                "input": body.text,
                                "output": outcome.output,
                                "last_agent": outcome.last_agent,
                                "events": outcome.events,
                                "pending": outcome.pending,
                            }
                        )
                        database.add_message(thread_id, "assistant", outcome.output, trace["id"])
                        await queue.put({"type": "done", "trace": trace, "outcome": outcome.as_dict()})
                    except Exception as exc:
                        await queue.put({"type": "error", "error": str(exc)})
                    finally:
                        await queue.put(None)

                task = asyncio.create_task(runner())
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield _sse(item)
                await task

            return StreamingResponse(events(), media_type="text/event-stream")
        return await _go()

    @app.get("/api/connectors")
    def connectors() -> list[dict[str, Any]]:
        return db().list_connectors()

    @app.post("/api/connectors")
    def create_connector(body: ConnectorBody) -> dict[str, Any]:
        try:
            return db().upsert_connector(body.model_dump())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.patch("/api/connectors/{connector_id}")
    def patch_connector(connector_id: str, body: ConnectorPatch) -> dict[str, Any]:
        try:
            current = db().get_connector(connector_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        changes = body.model_dump(exclude_none=True)
        current.update(changes)
        return db().upsert_connector(current)

    @app.delete("/api/connectors/{connector_id}")
    def delete_connector(connector_id: str) -> dict[str, bool]:
        try:
            db().delete_connector(connector_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True}

    @app.get("/api/datasets")
    def datasets() -> list[dict[str, Any]]:
        return db().list_datasets()

    @app.post("/api/datasets")
    def create_dataset(body: DatasetBody) -> dict[str, Any]:
        return db().create_dataset(body.name, body.tasks, body.workflow_id)

    @app.get("/api/evals")
    def evals() -> list[dict[str, Any]]:
        return db().list_eval_runs()

    if FRONTEND_DIST.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            candidate = FRONTEND_DIST / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app


_STORE: Store | None = None


def _default_store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
    return _STORE


def _or_404(fn: Any) -> Any:
    try:
        return fn()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def load_document_text(text: str) -> WorkflowDocument:
    import yaml

    data = yaml.safe_load(text)
    return load_document(data)


app = create_app()
