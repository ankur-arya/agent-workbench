"""Input and output checks used by guardrail nodes and SDK guardrails.

These are deterministic stand-ins for the hosted guardrails that mask or flag
PII, jailbreaks, and a few output-policy failures. They do not call a model.
"""

from __future__ import annotations

import re

EMAIL = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE = re.compile(r"\b(?:\+?\d{1,2}[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]?\d{3}[\s.\-]?\d{4}\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9]{8,}|api[_-]?key\s*[:=]\s*\S+)\b", re.IGNORECASE)

JAILBREAKS = (
    "ignore previous instructions",
    "ignore all previous",
    "ignore your instructions",
    "you are now",
    "reveal your system prompt",
    "show your system prompt",
    "jailbreak",
    "do anything now",
    "developer mode",
)


def mask_pii(text: str) -> str:
    masked = EMAIL.sub("[email]", text)
    masked = SSN.sub("[ssn]", masked)
    masked = PHONE.sub("[phone]", masked)
    return masked


def findings_for(text: str, checks: list[str]) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    lowered = text.lower()
    if "pii" in checks:
        if EMAIL.search(text):
            found.append({"check": "pii", "detail": "email address"})
        if PHONE.search(text):
            found.append({"check": "pii", "detail": "phone number"})
        if SSN.search(text):
            found.append({"check": "pii", "detail": "SSN"})
    if "jailbreak" in checks:
        for phrase in JAILBREAKS:
            if phrase in lowered:
                found.append({"check": "jailbreak", "detail": phrase})
                break
    if "keywords" in checks:
        found.extend(
            {"check": "keywords", "detail": word}
            for word in ("explosive", "malware", "phishing")
            if word in lowered
        )
    if "secrets" in checks and SECRET.search(text):
        found.append({"check": "secrets", "detail": "secret-like token"})
    if "empty" in checks and not text.strip():
        found.append({"check": "empty", "detail": "empty output"})
    return found
