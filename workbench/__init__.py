"""Local-first workbench for building agents on the OpenAI Agents SDK."""

from agents import set_tracing_disabled

# Never export traces unless a run explicitly turns tracing on and a key is present.
set_tracing_disabled(True)

__version__ = "0.1.0"

from workbench.document import HarnessSpec, WorkflowDocument
from workbench.harness import run_agent
from workbench.runtime import run_workflow
from workbench.templates import get_template, list_templates
from workbench.yamlio import load_document

__all__ = [
    "HarnessSpec",
    "WorkflowDocument",
    "__version__",
    "get_template",
    "list_templates",
    "load_document",
    "run_agent",
    "run_workflow",
]
