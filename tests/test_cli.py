from pathlib import Path

from workbench.cli import main
from workbench.templates import get_template
from workbench.yamlio import dump_yaml, load_document


def test_cli_demo_eval_and_yaml_roundtrip(tmp_path: Path, capsys) -> None:
    assert main(["--home", str(tmp_path), "demo"]) == 0
    assert "4" in capsys.readouterr().out

    assert main(["--home", str(tmp_path), "eval", "--template", "triage"]) == 0
    assert "2/2 passed" in capsys.readouterr().out

    config = tmp_path / "agent.yaml"
    config.write_text(dump_yaml(get_template("tool_assistant")), encoding="utf-8")
    loaded = load_document(config)
    assert loaded.name == "Tool assistant"
    assert main(["--home", str(tmp_path), "run", "--config", str(config), "--input", "What is 2 + 2?"]) == 0
    assert "4" in capsys.readouterr().out

    assert main(["--home", str(tmp_path), "init", "--template", "blank", "--name", "From CLI"]) == 0
    created = capsys.readouterr().out
    assert "wf_" in created

    assert main(["--home", str(tmp_path), "export", "--template", "triage", "--format", "python"]) == 0
    exported = capsys.readouterr().out
    assert "handoffs=" in exported

    assert main(["--home", str(tmp_path), "info"]) == 0
    info = capsys.readouterr().out
    assert "tool_assistant" in info
    assert "not set" in info


def test_custom_harness_registration() -> None:
    from workbench.document import HarnessSpec
    from workbench.harness import Harness, HarnessResult, execute, register_harness

    class EchoHarness(Harness):
        name = "echo"

        async def arun(self, agent, user_input, spec, **kwargs):
            return HarnessResult(status="completed", output=f"echo:{user_input}", last_agent=agent.name)

    register_harness(EchoHarness())
    agent_doc = __import__("workbench").get_template("blank")
    agent_doc.harness = HarnessSpec(kind="echo")
    outcome = __import__("workbench").run_workflow(agent_doc, "ping")
    assert outcome.output == "echo:ping"

    import asyncio
    from workbench.build import define_agent

    result = asyncio.run(execute(define_agent(name="A", instructions="x"), "ping", HarnessSpec(kind="echo")))
    assert result.output == "echo:ping"
