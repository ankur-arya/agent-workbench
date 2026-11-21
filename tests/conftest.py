from __future__ import annotations

from pathlib import Path

import pytest

from workbench.harness import clear_memory_sessions


@pytest.fixture(autouse=True)
def workbench_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("WORKBENCH_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    clear_memory_sessions()
    yield tmp_path
    clear_memory_sessions()
