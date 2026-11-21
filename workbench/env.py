"""Environment loading. Secrets stay in the environment or a gitignored .env file."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_LOADED = False


def load_env() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def openai_api_key() -> str | None:
    load_env()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key or None


def openai_model() -> str:
    load_env()
    return os.environ.get("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"


def connector_token(connector_id: str) -> str | None:
    load_env()
    env_name = "AGENT_WORKBENCH_" + connector_id.upper().replace("-", "_") + "_TOKEN"
    token = os.environ.get(env_name, "").strip()
    return token or None
