from decimal import Decimal

import pytest
from agents.usage import Usage

from workbench.pricing import (
    can_afford,
    estimate_task_usd,
    money,
    price_tokens,
    price_web_search,
    rates_for,
    usage_from_sdk,
)
from workbench.runtime import run_workflow
from workbench.templates import get_template


def test_rates_match_the_pricing_page_and_snapshots() -> None:
    assert rates_for("gpt-4o").input == Decimal("2.50")
    assert rates_for("gpt-4o").cached_input == Decimal("1.25")
    assert rates_for("gpt-4o").output == Decimal("10.00")
    assert rates_for("gpt-4o-2024-08-06").input == Decimal("2.50")
    assert rates_for("gpt-4o-mini").input == Decimal("0.15")
    assert rates_for("gpt-4o-mini-2024-07-18").cached_input == Decimal("0.075")
    assert rates_for("gpt-4.1").output == Decimal("8.00")
    assert rates_for("gpt-4.1-mini").cached_input == Decimal("0.10")
    with pytest.raises(KeyError):
        rates_for("gpt-5")


def test_cached_tokens_are_not_billed_again_as_full_input() -> None:
    cached_only = price_tokens("gpt-4o", input_tokens=1_000_000, cached_input_tokens=1_000_000, output_tokens=0)
    assert cached_only == Decimal("1.25")
    mixed = price_tokens("gpt-4o", input_tokens=1_000_000, cached_input_tokens=400_000, output_tokens=0)
    assert mixed == Decimal("2.00")
    assert price_tokens("gpt-4o-mini", input_tokens=0, cached_input_tokens=0, output_tokens=1_000_000) == Decimal("0.60")
    assert price_web_search(1, billable=True) == Decimal("0.01")
    assert price_web_search(3, billable=False) == Decimal(0)
    assert money(Decimal("0.01")) == "0.010000"


def test_usage_from_sdk_reads_context_usage() -> None:
    usage = Usage(requests=2, input_tokens=1500, output_tokens=250, total_tokens=1750)
    usage.input_tokens_details.cached_tokens = 400
    parsed = usage_from_sdk(usage)
    assert parsed == {
        "requests": 2,
        "input_tokens": 1500,
        "cached_input_tokens": 400,
        "output_tokens": 250,
    }


def test_local_agent_run_exposes_sdk_usage_without_a_price() -> None:
    outcome = run_workflow(get_template("web_search_agent"), "steve jobs statue location budapst")
    assert outcome.status == "completed"
    assert outcome.usage is not None
    assert outcome.usage["requests"] >= 1
    assert outcome.usage["cached_input_tokens"] == 0
    assert any(event.get("name") == "web_search" for event in outcome.events if event.get("type") == "tool")


def test_estimate_is_zero_for_the_local_provider_and_positive_for_gpt_4o() -> None:
    assert estimate_task_usd(agent_model="local", judge_model=None, max_turns=10, bill_web_search=False) == 0
    estimate = estimate_task_usd(
        agent_model="gpt-4o",
        judge_model="gpt-4o-mini",
        max_turns=10,
        bill_web_search=True,
    )
    assert estimate > Decimal("0.50")
    assert can_afford(Decimal(0), estimate, estimate)
    assert not can_afford(Decimal("0.01"), estimate, estimate)
