# Agent workbench

**Build. Preview. Evaluate.**

A local-first workbench for designing agent workflows on the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/). You start from a template, compose nodes on a canvas, pick a harness, preview the agent in chat, and publish a version when the evals look right.

The shape follows OpenAI’s AgentKit surfaces (Agent Builder, ChatKit, connectors, guardrails, evals) and runs them on the open-source SDK, with a deterministic local model so the whole loop works without an API key. Hosted Agent Builder and the hosted Evals product are scheduled to shut down on November 30, 2026. This workbench is the local stand-in for that workflow.

Day-1 path: `pip install` + one command. SQLite + files on disk. No Docker.

## Quick start (no Docker)

```bash
git clone https://github.com/ankur-arya/agent-workbench.git
cd agent-workbench

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Starts API + UI. First launch builds the React SPA (needs Node 18+).
python -m workbench ui
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Click **Tool assistant**, then in Preview send `What is 2 + 2?`. The local model calls the calculator and answers `4`. No `OPENAI_API_KEY` is required.

`uv` works the same way:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python -m workbench ui
```

Makefile equivalents: `make install`, `make ui`, `make demo`, `make eval`, `make test`.

## CLI

```bash
python -m workbench info
python -m workbench templates

# One shot
python -m workbench demo
python -m workbench run --template tool_assistant --input "What is 2 + 2?"

# Grade the sample tasks that ship with a template
python -m workbench eval --template triage

# Terminal preview (session memory on by default)
python -m workbench chat --template research

# Save a template, snapshot it, print Agents SDK Python
python -m workbench init --template homework_helper
python -m workbench publish --workflow wf_…
python -m workbench export --template orchestrator --format python

python -m workbench connectors
```

`run` also accepts `--config configs/tool_assistant.yaml`, `--workflow`, `--version`, `--harness`, `--provider local|openai`, and `--session`.

Copy `.env.example` to `.env` only when a harness uses `provider: openai`. The key is read from the environment and is never written into a workflow.

## What you get

| Surface | Role |
| --- | --- |
| **Workflows** | Templates and saved drafts. Each card shows harness and version. |
| **Studio** | Canvas, node inspector, harness picker, preview chat, traces, evals, code export. |
| **Connectors** | Built-in tools, legacy connector rows, and MCP servers you register. |
| **Datasets** | Sample tasks and the eval runs that grade them. |

### Templates

| Id | What it shows |
| --- | --- |
| `tool_assistant` | One agent, calculator and clock. |
| `research` | Local handbook and web index search. |
| `triage` | Classifier hands off to a math tutor or a general assistant. |
| `orchestrator` | Specialists exposed as tools (`Agent.as_tool`). |
| `guardrailed` | PII mask and a keyword block on the way in and out. |
| `homework_helper` | Rewrite, classify, then route. The Agent Builder homework example, locally. |
| `review` | Human-approval pause before a note is sent. |
| `blank` | Start node and one empty agent. |

### Harnesses

| Kind | What it does |
| --- | --- |
| `sync` | `Runner.run_sync`. One final result. |
| `async` | Same loop, awaited. |
| `streaming` | `Runner.run_streamed`. The preview shows token deltas. |
| `approval` | Pause, auto-approve, or reject when a tool needs a person. |
| `eval` | One task inside a graded sweep. Sessions are turned off so tasks stay independent. |

Other harness knobs: `provider` (`local` or `openai`), `max_turns`, `tool_choice`, `tool_use_behavior` (`run_llm_again` or `stop_on_first_tool`), `session_backend` (`none`, `memory`, `sqlite`), `approval` (`manual`, `auto`, `reject`), and `tracing`.

Chat threads force a sqlite session keyed by the thread. Tracing stays off unless the harness enables it, the provider is `openai`, and `OPENAI_API_KEY` is set. The package disables SDK trace export at import so tests never phone home.

## Local storage

Override the data directory with `WORKBENCH_HOME` (or `--home`). Default is `./.workbench`.

| Path | Role |
| --- | --- |
| `.workbench/workbench.db` | Workflows, versions, threads, traces, connectors, datasets, eval runs |
| `.workbench/sessions.db` | SDK `SQLiteSession` transcripts |
| `.workbench/` | Created on first use and gitignored |

## Architecture

```
CLI / React studio
        │
        ▼
   FastAPI (workbench.api)
        │
        ├── Store (SQLite)          drafts, published versions, traces
        ├── Templates               WorkflowDocument builders
        └── Runtime                 walks the canvas (cap 40 steps)
                │
                ├── logic nodes     if/else, while, transform, set_state,
                │                   guardrails, human approval, file search, MCP
                └── agent nodes ──► Harness ──► Agents SDK Runner
                                        │
                                        ├── LocalModel   deterministic, no key
                                        └── OpenAI       needs OPENAI_API_KEY
```

A workflow is a graph of nodes and edges, not one flattened agent. Agent nodes become SDK `Agent`s (instructions, tools, handoffs, input/output guardrails, structured output). Logic nodes run in the workbench. Expressions are a small language — `contains`, `not_contains`, `equals`, `empty`, `not_empty`, `word_count < N` — so the canvas can branch without `eval`.

Agent input: a `query` state variable if one was set, otherwise the previous output, otherwise `input_as_text` from the start node.

The React + Vite SPA is the product UI. `workbench ui` builds `frontend/dist` when it is missing and serves it from the same process as the API.

This matches the sibling workbenches: a `workbench` package, `python -m workbench`, FastAPI, a dark/light React SPA, YAML configs, pytest, and a local SQLite store. [ml-workbench](https://github.com/ankur-arya/ml-workbench) is the closer layout (CLI + studio + store). [ai-workbench](https://github.com/ankur-arya/ai-workbench) contributes the offline path: a local provider so demos run with no key, and `.env.example` for the optional OpenAI provider. Neither sibling ships GitHub Actions, so this repo does not either.

## Adding an agent type

A template is a function that returns a `WorkflowDocument`. Register it in `TEMPLATES` in `workbench/templates.py`:

```python
def desk(name: str = "") -> WorkflowDocument:
    return _doc(
        name or "Desk",
        "Answers with the calculator.",
        [
            Node(id="start", type="start", x=40, y=160),
            Node(id="assistant", type="agent", x=300, y=120, config={
                "name": "Desk",
                "instructions": "Use the calculator for arithmetic.",
                "tools": ["calculator"],
            }),
        ],
        [Edge(id="e1", source="start", target="assistant")],
        tasks=[EvalTask(input="What is 2 + 2?", expect_contains="4", expect_tool="calculator")],
    )

TEMPLATES["desk"] = desk
```

YAML is the other declarative path. `configs/` holds dumped templates, and `python -m workbench run --config configs/custom_agent.yaml` loads one. `examples/custom_agent.py` builds a single agent in code with `define_agent` and `run_agent`.

Edges between agents mean different things:

- A plain edge runs the next node in the graph.
- `handoff` makes the target a specialist the SDK can transfer to.
- `tool` makes the target an agent-as-tool, a file-search node, or an MCP node.

## Adding a harness

Subclass `Harness`, set `name`, override `arun`, and call `register_harness`. `execute` delegates to `arun` only when the subclass actually overrides it. Built-in kinds share one SDK loop and differ by mode.

```python
from workbench.document import HarnessSpec
from workbench.harness import Harness, HarnessResult, register_harness
from workbench.runtime import run_workflow
from workbench.templates import get_template

class EchoHarness(Harness):
    name = "echo"

    async def arun(self, agent, user_input, spec, **kwargs):
        return HarnessResult(status="completed", output=f"echo:{user_input}", last_agent=agent.name)

register_harness(EchoHarness())
document = get_template("blank")
document.harness = HarnessSpec(kind="echo")
print(run_workflow(document, "ping").output)  # echo:ping
```

`HarnessSpec.kind` is a string, so a custom name validates.

## OpenAI enterprise features

Checked against OpenAI’s docs on October 22, 2026. Agent Builder and the hosted Evals product shut down on November 30, 2026. New work is steered to the Agents SDK and self-hosted ChatKit. This table is the product contract for v1.

| OpenAI surface | In agent-workbench | Deferred |
| --- | --- | --- |
| [Agent Builder](https://developers.openai.com/api/docs/guides/agent-builder) canvas, templates, preview, publish, versions | Studio canvas, eight templates, preview chat, **Publish** snapshots a version, the Versions menu restores one | Hosted workflow IDs and the OpenAI canvas itself |
| [Node reference](https://developers.openai.com/api/docs/guides/node-reference): start, agent, note, file search, guardrails, MCP, if/else, while, human approval, transform, set state | All ten node types. Start exposes `input_as_text`. Notes do not run. | Full CEL. Expressions are the small language above. Hosted vector stores. |
| [ChatKit](https://developers.openai.com/api/docs/guides/chatkit) embed | Preview panel: threads, streaming SSE, tool and handoff lines, approve / reject | `@openai/chatkit-react`, widgets, and hosted ChatKit sessions. [Custom ChatKit](https://developers.openai.com/api/docs/guides/custom-chatkit) is the migration path if you embed later. |
| Code export ([migrate from Agent Builder](https://developers.openai.com/api/docs/guides/agent-builder/migrate-from-agent-builder)) | **Code** tab and `export --format python` emit Agents SDK Python | TypeScript export. Logic nodes are comments, because the SDK export does not carry the graph. |
| [Connector Registry](https://openai.com/index/introducing-agentkit/) and [MCP / legacy connectors](https://developers.openai.com/api/docs/guides/tools-connectors-mcp) | Local registry: builtins, disabled legacy rows (Dropbox, Gmail, Drive, Calendar, Teams, Outlook, SharePoint), and MCP URLs you add. OpenAI provider + `AGENT_WORKBENCH_<ID>_TOKEN` attaches `HostedMCPTool`. Local provider uses an MCP stub. | Global Admin Console, OAuth to those SaaS apps, and the ChatGPT Apps admin rename. `connector_id` is deprecated for models after September 1, 2026; prefer `server_url`. |
| [Guardrails](https://openai.com/index/introducing-agentkit/) (PII, jailbreak, hallucination; mask or flag) | Canvas guardrail node plus SDK input/output guardrails on the agent: email, phone, SSN, secrets, jailbreak phrases, keyword block. Flag trips the run. Mask rewrites. | The hosted guardrails library and a model-based hallucination check. |
| [Evals and trace grading](https://developers.openai.com/api/docs/guides/agent-evals) | Datasets, **Run eval**, and per-trace **Grade**. Graders: contains, exact, tool, handoff, guardrail, status, Jaccard similarity, word count. | Score/label model graders, the prompt optimizer, third-party model evals, and the hosted Evals API (winding down with Agent Builder). |
| Agents SDK tracing | Every run is stored locally and shown in **Traces** | Export to OpenAI Logs only when tracing is on, the provider is `openai`, and a key is set |
| [Reinforcement fine-tuning](https://openai.com/index/introducing-agentkit/) and [ChatGPT Workspace Agents](https://developers.openai.com/api/docs/guides/agent-builder/migrate-from-agent-builder) | — | Both stay on OpenAI’s platform |

Sources:

- [Introducing AgentKit](https://openai.com/index/introducing-agentkit/) (October 6, 2025; shutdown note June 3, 2026)
- [Agent Builder](https://developers.openai.com/api/docs/guides/agent-builder)
- [Node reference](https://developers.openai.com/api/docs/guides/node-reference)
- [ChatKit](https://developers.openai.com/api/docs/guides/chatkit)
- [Custom ChatKit](https://developers.openai.com/api/docs/guides/custom-chatkit)
- [Migrate from Agent Builder](https://developers.openai.com/api/docs/guides/agent-builder/migrate-from-agent-builder)
- [MCP and connectors](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)
- [Evaluate agent workflows](https://developers.openai.com/api/docs/guides/agent-evals)
- [Trace grading](https://developers.openai.com/api/docs/guides/trace-grading)
- [Getting started with evals](https://developers.openai.com/api/docs/guides/evaluation-getting-started)
- [Agents SDK guide](https://developers.openai.com/api/docs/guides/agents/sdk)
- [Agents SDK for Python](https://openai.github.io/openai-agents-python/)

## Tests

```bash
python -m pytest
```

Tests use the local model and a temporary `WORKBENCH_HOME`. `OPENAI_API_KEY` is unset. Coverage includes the expression language, every template’s sample tasks, session memory, tool approval, the human-approval node, the HTTP API (publish, restore, SSE, resume), the CLI, and a custom harness.

## Project layout

```
workbench/           SDK wiring, canvas runtime, store, FastAPI, CLI
frontend/            React + Vite SPA
configs/             YAML workflows
examples/            custom_agent.py
tests/               pytest
scripts/ui.sh        python -m workbench ui
```
