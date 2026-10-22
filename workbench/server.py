"""Start the API and, when Node is available, the built UI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"


def ensure_frontend_built(*, skip_build: bool = False) -> Path:
    dist = FRONTEND / "dist" / "index.html"
    if dist.exists() or skip_build:
        return FRONTEND / "dist"
    npm = shutil.which("npm")
    if npm is None:
        print("npm is not on PATH. Serving the API only. Install Node 18+ to build the UI.", file=sys.stderr)
        return FRONTEND / "dist"
    print("Building the workbench UI (first run)…")
    subprocess.check_call([npm, "install"], cwd=FRONTEND)
    subprocess.check_call([npm, "run", "build"], cwd=FRONTEND)
    return FRONTEND / "dist"


def serve(host: str = "127.0.0.1", port: int = 8000, *, reload: bool = False, skip_build: bool = False) -> None:
    ensure_frontend_built(skip_build=skip_build)
    import uvicorn

    from workbench.api import create_app

    uvicorn.run(
        "workbench.api:app" if reload else create_app(),
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
