"""OpenAI token and web-search prices for benchmark cost tracking.

Standard prices, USD per 1M tokens, from
https://developers.openai.com/api/docs/pricing
checked 2026-11-22.

| Model | Input | Cached input | Output |
| --- | --- | --- | --- |
| gpt-4o | 2.50 | 1.25 | 10.00 |
| gpt-4o-mini | 0.15 | 0.075 | 0.60 |
| gpt-4.1 | 2.00 | 0.50 | 8.00 |
| gpt-4.1-mini | 0.40 | 0.10 | 1.60 |

Web search (including images), the non-preview tool the Agents SDK
``WebSearchTool`` calls: $10.00 per 1,000 calls, which is $0.01 per call.
Search content tokens are billed at the model input rate and show up in the
response usage, so they are not added a second time here. For gpt-4o-mini and
gpt-4.1-mini, OpenAI bills that content as a fixed 8,000 input tokens per
non-preview web search call.

Cached input tokens are a subset of input tokens. Uncached input is
``input - cached``, billed at the input rate. Cached tokens are billed at the
cached-input rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
PRICING_CHECKED = "2026-11-22"

# Per 1M tokens. Cached input is the discounted rate for the cached subset.
_MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelRates:
    input: Decimal
    cached_input: Decimal
    output: Decimal


RATES: dict[str, ModelRates] = {
    "gpt-4o": ModelRates(Decimal("2.50"), Decimal("1.25"), Decimal("10.00")),
    "gpt-4o-mini": ModelRates(Decimal("0.15"), Decimal("0.075"), Decimal("0.60")),
    "gpt-4.1": ModelRates(Decimal("2.00"), Decimal("0.50"), Decimal("8.00")),
    "gpt-4.1-mini": ModelRates(Decimal("0.40"), Decimal("0.10"), Decimal("1.60")),
}

# $10.00 / 1,000 calls.
WEB_SEARCH_USD_PER_CALL = Decimal("0.01")

# Reservation for the next task. It is larger than a typical search-agent turn
# so a task is not started unless the remaining budget can cover it.
# max_turns requests, each with this many tokens, plus two hosted web searches
# and one judge call. Cached tokens are estimated as 0 because uncached input
# costs more on every model in this table.
ESTIMATE_INPUT_TOKENS_PER_REQUEST = 20_000
ESTIMATE_OUTPUT_TOKENS_PER_REQUEST = 2_000
ESTIMATE_WEB_SEARCH_CALLS = 2
ESTIMATE_JUDGE_INPUT_TOKENS = 4_000
ESTIMATE_JUDGE_OUTPUT_TOKENS = 500


def empty_usage() -> dict[str, int]:
    return {
        "requests": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
    }


def usage_from_sdk(usage: Any) -> dict[str, int]:
    """Read Agents SDK ``RunResult.context_wrapper.usage`` (or a ``Usage``)."""
    if usage is None:
        return empty_usage()
    details = getattr(usage, "input_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    return {
        "requests": int(getattr(usage, "requests", 0) or 0),
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": cached,
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }


def usage_from_chat_completion(usage: Any) -> dict[str, int]:
    """Read ``response.usage`` from a Chat Completions result."""
    if usage is None:
        return empty_usage()
    details = getattr(usage, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    return {
        "requests": 1,
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "cached_input_tokens": cached,
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in empty_usage()}


def rates_for(model: str) -> ModelRates:
    """Match a snapshot id such as ``gpt-4o-mini-2024-07-18`` to a priced model."""
    name = (model or "").strip()
    for key in sorted(RATES, key=len, reverse=True):
        if name == key or name.startswith(key + "-"):
            return RATES[key]
    known = ", ".join(sorted(RATES))
    raise KeyError(f"No price for model {model!r}. Known models: {known}")


def price_tokens(
    model: str,
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> Decimal:
    rates = rates_for(model)
    cached = min(max(int(cached_input_tokens), 0), max(int(input_tokens), 0))
    uncached = max(int(input_tokens), 0) - cached
    return (
        Decimal(uncached) * rates.input / _MILLION
        + Decimal(cached) * rates.cached_input / _MILLION
        + Decimal(max(int(output_tokens), 0)) * rates.output / _MILLION
    )


def price_web_search(calls: int, *, billable: bool) -> Decimal:
    if not billable or calls <= 0:
        return Decimal(0)
    return WEB_SEARCH_USD_PER_CALL * calls


def estimate_task_usd(
    *,
    agent_model: str,
    judge_model: str | None,
    max_turns: int,
    bill_web_search: bool,
) -> Decimal:
    """Upper-bound reservation checked before a task is started.

    Local runs pass no agent model and return 0. A priced model is reserved
    for ``max_turns`` requests so the harness cannot start a task whose worst
    case inside that token budget would cross the cap.
    """
    if not agent_model or agent_model == "local":
        return Decimal(0)
    requests = max(1, int(max_turns))
    agent = price_tokens(
        agent_model,
        input_tokens=ESTIMATE_INPUT_TOKENS_PER_REQUEST * requests,
        cached_input_tokens=0,
        output_tokens=ESTIMATE_OUTPUT_TOKENS_PER_REQUEST * requests,
    )
    search = price_web_search(ESTIMATE_WEB_SEARCH_CALLS, billable=bill_web_search)
    judge = Decimal(0)
    if judge_model:
        judge = price_tokens(
            judge_model,
            input_tokens=ESTIMATE_JUDGE_INPUT_TOKENS,
            cached_input_tokens=0,
            output_tokens=ESTIMATE_JUDGE_OUTPUT_TOKENS,
        )
    return agent + search + judge


def money(value: Decimal) -> str:
    """Fixed-point dollars, six places, so JSON does not use binary floats."""
    return format(value.quantize(Decimal("0.000001")), "f")


def can_afford(spent: Decimal, estimate: Decimal, budget: Decimal) -> bool:
    return spent + estimate <= budget
