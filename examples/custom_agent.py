"""Define an agent in code and run it on the local harness.

No API key is required. Set OPENAI_API_KEY and pass provider="openai"
on the harness when you want the real model.
"""

from workbench.build import define_agent
from workbench.document import HarnessSpec
from workbench.harness import run_agent


def main() -> None:
    agent = define_agent(
        name="Desk calculator",
        instructions="Use the calculator tool for arithmetic and answer with the result.",
        tools=["calculator"],
    )
    result = run_agent(agent, "What is (2 + 2) * 5?", HarnessSpec(kind="sync", provider="local"))
    print(result.output)
    print("status:", result.status)


if __name__ == "__main__":
    main()
