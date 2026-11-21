"""Load and save workflow documents as YAML or JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from workbench.document import WorkflowDocument


def load_document(source: str | Path | dict[str, Any]) -> WorkflowDocument:
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Workflow file must contain a mapping")
    return WorkflowDocument.model_validate(data)


def dump_yaml(document: WorkflowDocument) -> str:
    return yaml.safe_dump(document.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
