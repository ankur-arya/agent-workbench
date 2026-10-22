"""Command-line interface: templates, run, chat, eval, publish, ui."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from workbench.catalog import harness_catalog
from workbench.document import WorkflowDocument
from workbench.env import load_env, openai_api_key
from workbench.evals import run_eval
from workbench.export import export_python
from workbench.harness import clear_memory_sessions
from workbench.paths import workbench_home
from workbench.runtime import run_workflow
from workbench.store import Store
from workbench.templates import get_template, list_templates
from workbench.yamlio import dump_yaml, load_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="workbench",
        description="Local-first agent workbench on the OpenAI Agents SDK. No Docker.",
    )
    parser.add_argument("--home", default=None, help="Data directory (default: $WORKBENCH_HOME or ./.workbench).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Show templates, harnesses, and whether an API key is set.")
    sub.add_parser("templates", help="List agent templates.")

    init = sub.add_parser("init", help="Save a template into the local store.")
    init.add_argument("--template", default="tool_assistant")
    init.add_argument("--name", default=None)

    run = sub.add_parser("run", help="Run one input through a template, config, or stored workflow.")
    run.add_argument("--template", default=None)
    run.add_argument("--config", type=Path, default=None)
    run.add_argument("--workflow", default=None)
    run.add_argument("--version", default=None)
    run.add_argument("--input", required=True)
    run.add_argument("--harness", default=None, help="sync, async, streaming, approval, or eval.")
    run.add_argument("--provider", default=None, choices=["local", "openai"])
    run.add_argument("--session", default=None, help="Session id. Uses the sqlite backend when set.")

    chat = sub.add_parser("chat", help="Interactive preview in the terminal.")
    chat.add_argument("--template", default="tool_assistant")
    chat.add_argument("--config", type=Path, default=None)
    chat.add_argument("--harness", default=None)
    chat.add_argument("--session", default="cli")

    eval_p = sub.add_parser("eval", help="Grade a template or config against its sample tasks.")
    eval_p.add_argument("--template", default=None)
    eval_p.add_argument("--config", type=Path, default=None)
    eval_p.add_argument("--workflow", default=None)

    publish = sub.add_parser("publish", help="Snapshot a stored workflow as the next version.")
    publish.add_argument("--workflow", required=True)
    publish.add_argument("--note", default="Published from the CLI")

    sub.add_parser("connectors", help="List the connector registry.")

    export_p = sub.add_parser("export", help="Print YAML or Agents SDK Python for a template.")
    export_p.add_argument("--template", default="tool_assistant")
    export_p.add_argument("--format", choices=["yaml", "python"], default="yaml")

    ui = sub.add_parser("ui", help="Start the local API and UI.")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8000)
    ui.add_argument("--reload", action="store_true")
    ui.add_argument("--skip-build", action="store_true")

    sub.add_parser("demo", help="Run the tool-assistant sample and print the result.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.home:
        os.environ["WORKBENCH_HOME"] = str(Path(args.home).expanduser().resolve())
    if args.command == "info":
        return _info()
    if args.command == "templates":
        for item in list_templates():
            print(f"{item['id']:<18} {item['name']}")
            print(f"  {item['description']}")
        return 0
    if args.command == "init":
        document = get_template(args.template, args.name)
        workflow = Store().create_workflow(document)
        print(f"Created {workflow['id']}  {workflow['name']}")
        return 0
    if args.command == "run":
        return _run(args)
    if args.command == "chat":
        return _chat(args)
    if args.command == "eval":
        return _eval(args)
    if args.command == "publish":
        version = Store().publish(args.workflow, args.note)
        print(f"Published {version['workflow_id']} v{version['version']} ({version['id']})")
        return 0
    if args.command == "connectors":
        for connector in Store().list_connectors():
            state = "on" if connector["enabled"] else "off"
            print(f"{state:<4} {connector['kind']:<8} {connector['id']:<28} {connector['name']}")
        return 0
    if args.command == "export":
        document = get_template(args.template)
        print(export_python(document) if args.format == "python" else dump_yaml(document), end="")
        return 0
    if args.command == "ui":
        from workbench.server import serve

        serve(args.host, args.port, reload=args.reload, skip_build=args.skip_build)
        return 0
    if args.command == "demo":
        document = get_template("tool_assistant")
        outcome = run_workflow(document, "What is 2 + 2?")
        print(outcome.output)
        print(f"status={outcome.status} agent={outcome.last_agent}")
        return 0 if outcome.status == "completed" else 1
    parser.error(f"Unknown command {args.command}")
    return 2


def _info() -> int:
    print(f"Home: {workbench_home()}")
    print(f"OpenAI key: {'set' if openai_api_key() else 'not set (local provider will be used)'}")
    print("Templates:")
    for item in list_templates():
        print(f"  {item['id']}")
    print("Harnesses:")
    for item in harness_catalog():
        print(f"  {item['kind']:<12} {item['summary']}")
    return 0


def _load(args: argparse.Namespace) -> WorkflowDocument:
    if getattr(args, "config", None):
        document = load_document(args.config)
    elif getattr(args, "workflow", None):
        document = Store().document_for(args.workflow, getattr(args, "version", None))
    else:
        document = get_template(getattr(args, "template", None) or "tool_assistant")
    if getattr(args, "harness", None):
        document.harness.kind = args.harness
    if getattr(args, "provider", None):
        document.harness.provider = args.provider
    if getattr(args, "session", None):
        document.harness.session_backend = "sqlite"
        document.harness.session_id = args.session
    return document


def _run(args: argparse.Namespace) -> int:
    document = _load(args)
    outcome = run_workflow(document, args.input)
    print(outcome.output)
    print(f"status={outcome.status} harness={outcome.harness} agent={outcome.last_agent}")
    if outcome.events:
        print(json.dumps(outcome.events, indent=2))
    return 0 if outcome.status in {"completed", "needs_approval", "blocked"} else 1


def _chat(args: argparse.Namespace) -> int:
    document = _load(args)
    if document.harness.session_backend == "none":
        document.harness.session_backend = "memory"
        document.harness.session_id = args.session
    print(f"{document.name}  ({document.harness.kind}, {document.harness.provider}). Empty line exits.")
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            print()
            return 0
        if not text:
            return 0
        outcome = run_workflow(document, text)
        print(f"agent> {outcome.output}")
        if outcome.status == "needs_approval" and outcome.pending:
            decision = input("approve? [y/n] ").strip().lower()
            resumed = run_workflow(
                document,
                text,
                resume={**outcome.pending, "decision": "approve" if decision.startswith("y") else "reject"},
            )
            print(f"agent> {resumed.output}")
    clear_memory_sessions()
    return 0


def _eval(args: argparse.Namespace) -> int:
    document = _load(args)
    outcome = _run_eval_sync(document)
    report = outcome.report or {}
    print(outcome.output)
    for row in report.get("results") or []:
        mark = "pass" if row["passed"] else "FAIL"
        print(f"  [{mark}] {row['input']}")
        if not row["passed"]:
            print(f"         {row['output'][:180]}")
    return 0 if report.get("failed", 1) == 0 else 1


def _run_eval_sync(document: WorkflowDocument):
    import asyncio

    return asyncio.run(run_eval(document))
