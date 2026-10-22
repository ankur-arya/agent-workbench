"""Local data directory. Override with WORKBENCH_HOME."""

from __future__ import annotations

import os
from pathlib import Path


def workbench_home() -> Path:
    raw = os.environ.get("WORKBENCH_HOME")
    path = Path(raw).expanduser() if raw else Path.cwd() / ".workbench"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()
