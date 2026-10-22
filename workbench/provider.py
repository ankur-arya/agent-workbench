"""Model providers.

`local` is a deterministic Agents SDK `Model`. It lets every template, harness,
and test run with no API key. `openai` uses the SDK's default provider and the
key in `OPENAI_API_KEY`.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from agents import ModelSettings, RunConfig
from agents.items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from agents.model_settings import ModelSettings as ModelSettingsType
from agents.models.interface import Model, ModelProvider, ModelTracing
from agents.models.openai_provider import OpenAIProvider
from agents.testing.model import ScriptedModel, assistant_message, function_call
from agents.usage import Usage
from openai.types.responses.response_prompt_param import ResponsePromptParam

from workbench.document import HarnessSpec
from workbench.env import openai_api_key, openai_model

_ARITH = re.compile(r"(\d+(?:\.\d+)?(?:\s*[-+*/%]\s*\d+(?:\.\d+)?)+)")
_CALL = 0


class LocalModel(Model):
    """Keyword router that emits real tool calls, handoffs, and final messages."""

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettingsType,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        del tracing, previous_response_id, conversation_id, prompt
        output = decide(
            system_instructions or "",
            input,
            tools,
            handoffs,
            output_schema,
            model_settings,
        )
        return ModelResponse(output=output, usage=Usage(requests=1), response_id=_response_id())

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettingsType,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        scripted = ScriptedModel([])

        async def _events() -> AsyncIterator[TResponseStreamEvent]:
            response = await self.get_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            )
            scripted.enqueue(response)
            async for event in scripted.stream_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            ):
                yield event

        return _events()


class LocalProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        del model_name
        return LocalModel()


def decide(
    instructions: str,
    model_input: str | list[TResponseInputItem],
    tools: list[Any],
    handoffs: list[Any],
    output_schema: Any,
    model_settings: ModelSettingsType,
) -> list[Any]:
    user_text = _user_text(model_input)
    called = _called_tools(model_input)
    tool_outputs = _tool_outputs(model_input)
    choice = getattr(model_settings, "tool_choice", None) or "auto"

    if output_schema is not None and not output_schema.is_plain_text() and not tools and not handoffs:
        return [assistant_message(_structured(output_schema, user_text, instructions))]

    if choice != "none" and handoffs and not tool_outputs:
        picked = _best_handoff(user_text, handoffs)
        if picked is not None:
            return [function_call(picked.tool_name, "{}", call_id=_call_id())]

    if choice != "none" and tools:
        pending = [tool for tool in tools if getattr(tool, "name", None) not in called]
        picked_tool = _best_tool(
            user_text,
            instructions,
            pending,
            required=choice == "required" and not called,
            allow_named=not called,
        )
        if picked_tool is not None:
            arguments = _arguments_for(picked_tool, user_text)
            return [function_call(picked_tool.name, arguments, call_id=_call_id())]

    if output_schema is not None and not output_schema.is_plain_text():
        return [assistant_message(_structured(output_schema, user_text, instructions, tool_outputs))]
    if tool_outputs:
        return [assistant_message(_final_from_tools(user_text, instructions, tool_outputs))]
    return [assistant_message(_prose(user_text, instructions, tool_outputs, earlier=_earlier_user_text(model_input)))]


def build_run_config(
    spec: HarnessSpec,
    *,
    workflow_name: str,
    model_override: Model | None = None,
) -> RunConfig:
    """Build the SDK RunConfig for one harness invocation."""
    from agents import set_tracing_disabled

    provider = spec.provider
    if model_override is not None:
        model: Model | str | None = model_override
        model_provider: ModelProvider = LocalProvider()
    elif provider == "local":
        model = LocalModel()
        model_provider = LocalProvider()
    else:
        if not openai_api_key():
            raise RuntimeError(
                "provider is openai but OPENAI_API_KEY is not set. "
                "Export the key or switch the harness provider to local."
            )
        model = spec.model or openai_model()
        model_provider = OpenAIProvider(api_key=openai_api_key())

    tracing_on = bool(spec.tracing and provider == "openai" and openai_api_key())
    set_tracing_disabled(not tracing_on)
    settings = ModelSettings(
        temperature=spec.temperature,
        tool_choice=spec.tool_choice,
        parallel_tool_calls=spec.parallel_tool_calls,
    )
    return RunConfig(
        model=model,
        model_provider=model_provider,
        model_settings=settings,
        tracing_disabled=not tracing_on,
        workflow_name=workflow_name or "agent-workbench",
        trace_include_sensitive_data=False,
    )


def _structured(
    output_schema: Any,
    user_text: str,
    instructions: str,
    tool_outputs: list[str] | None = None,
) -> str:
    schema = output_schema.json_schema()
    properties = schema.get("properties") or {}
    route = _route_label(user_text, instructions)
    payload: dict[str, Any] = {}
    for name, spec in properties.items():
        kind = spec.get("type")
        if name == "route":
            payload[name] = route
        elif kind == "string":
            payload[name] = _prose(user_text, instructions, tool_outputs or [])
        elif kind == "integer":
            payload[name] = 0
        elif kind == "number":
            payload[name] = 0
        elif kind == "boolean":
            payload[name] = True
        else:
            payload[name] = route if name == "route" else ""
    return json.dumps(payload)


def _route_label(user_text: str, instructions: str) -> str:
    haystack = f"{user_text} {instructions}".lower()
    if any(word in user_text.lower() for word in ("policy", "handbook", "refund", "vacation", "file")):
        return "facts"
    if "route=facts" in instructions and "facts" in haystack and "policy" in user_text.lower():
        return "facts"
    return "qa"


def _prose(user_text: str, instructions: str, tool_outputs: list[str], earlier: str = "") -> str:
    if tool_outputs:
        return _final_from_tools(user_text, instructions, tool_outputs)
    lowered = user_text.lower()
    if earlier and any(phrase in lowered for phrase in ("what did i", "just ask", "did i ask", "previous question")):
        return f"You asked: {earlier}"
    if "first president" in lowered:
        return "The first president of the United States was George Washington."
    if "photosynthesis" in lowered:
        return "Photosynthesis turns light, water, and carbon dioxide into sugar and oxygen."
    if "rewrite" in instructions.lower():
        cleaned = " ".join(user_text.split())
        return cleaned or "Please share a question."
    if _ARITH.search(user_text):
        from workbench.tools import calculator

        expression = _ARITH.search(user_text).group(1)
        return f"The result is {calculator(expression)}."
    if "password" in lowered:
        return (
            "To reset a password, open the account page and follow the emailed link. "
            "Links expire after one hour."
        )
    if "plan" in lowered and "paris" in lowered:
        return "One-day Paris plan: morning at the Musée d'Orsay, afternoon along the Seine, evening in the Marais."
    if "weather" in lowered and "paris" in lowered:
        return "Paris weather in October is mild, often 12 to 18 degrees, with occasional rain."
    snippet = user_text.strip() or "your question"
    return f"Here is a concise answer about {snippet}."


def _final_from_tools(user_text: str, instructions: str, tool_outputs: list[str]) -> str:
    joined = " ".join(tool_outputs).strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", joined):
        return f"The result is {joined}."
    if "rewrite" in instructions.lower():
        return user_text.strip()
    return joined or _prose(user_text, instructions, [])


def _best_handoff(user_text: str, handoffs: list[Any]) -> Any | None:
    text = user_text.lower()
    best = None
    best_score = 0
    for handoff in handoffs:
        description = f"{handoff.agent_name} {handoff.tool_description}".lower()
        score = _score(text, description)
        if any(token in text for token in ("*", "/", "+", "times", "plus")) and "math" in description:
            score += 3
        if "president" in text or "history" in text or "war" in text:
            if "history" in description:
                score += 3
        if score > best_score:
            best = handoff
            best_score = score
    return best if best_score > 0 else None


def _best_tool(
    user_text: str,
    instructions: str,
    tools: list[Any],
    *,
    required: bool,
    allow_named: bool = True,
) -> Any | None:
    if not tools:
        return None
    # Score the question, not the instructions. Instruction text names every
    # attached tool, and using it here made the clock fire on arithmetic.
    user = user_text.lower()
    hinted = instructions.lower()
    scored: list[tuple[int, Any]] = []
    for tool in tools:
        name = getattr(tool, "name", "")
        description = f"{name} {getattr(tool, 'description', '')}".lower()
        score = _score(user, description)
        if name == "calculator" and _ARITH.search(user_text):
            score += 5
        if name == "file_search" and any(word in user for word in ("handbook", "policy", "refund", "vacation", "password")):
            score += 4
        if name == "web_search" and any(word in user for word in ("weather", "paris", "web", "online", "research")):
            score += 4
        if name == "clock" and any(word in user for word in ("time", "date", "clock")):
            score += 4
        if name == "lookup_order" and "order" in user:
            score += 4
        if name == "send_note" and any(word in user for word in ("send", "email", "note")):
            score += 3
        if name == "take_note" and "remember" in user:
            score += 3
        if name.startswith("mcp_") and "connector" in user:
            score += 2
        scored.append((score, tool))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] > 0:
        return scored[0][1]
    if required:
        return tools[0]
    if not allow_named:
        return None
    for tool in tools:
        if tool.name in {"file_search", "web_search"} and "search" in hinted:
            return tool
    named = [tool for tool in tools if tool.name in hinted or tool.name.replace("_", " ") in hinted]
    if len(named) == 1:
        return named[0]
    return None


def _arguments_for(tool: Any, user_text: str) -> dict[str, Any]:
    schema = getattr(tool, "params_json_schema", {}) or {}
    properties = list((schema.get("properties") or {}).keys())
    expression = _ARITH.search(user_text)
    values: dict[str, Any] = {}
    for name in properties:
        if name in {"expression", "expr"} and expression:
            values[name] = expression.group(1).replace(" ", "")
        elif name in {"order_id", "id"}:
            match = re.search(r"\b\d{3,}\b", user_text)
            values[name] = match.group(0) if match else "1001"
        else:
            values[name] = user_text.strip() or "query"
    return values


def _score(text: str, description: str) -> int:
    score = 0
    for token in set(re.findall(r"[a-z0-9_]+", description)):
        if len(token) < 4:
            continue
        if token in text:
            score += 1
    return score


def _user_texts(model_input: str | list[TResponseInputItem]) -> list[str]:
    if isinstance(model_input, str):
        return [model_input]
    texts: list[str] = []
    for item in model_input:
        if isinstance(item, str):
            texts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role == "user" and isinstance(content, str):
            texts.append(content)
        elif role == "user" and isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"input_text", "text", "output_text"}:
                    texts.append(str(part.get("text") or ""))
    return texts


def _user_text(model_input: str | list[TResponseInputItem]) -> str:
    texts = _user_texts(model_input)
    return texts[-1] if texts else ""


def _earlier_user_text(model_input: str | list[TResponseInputItem]) -> str:
    texts = _user_texts(model_input)
    return " ".join(text for text in texts[:-1] if text).strip()


def _call_names(model_input: str | list[TResponseInputItem]) -> dict[str, str]:
    names: dict[str, str] = {}
    if isinstance(model_input, str):
        return names
    for item in model_input:
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            if isinstance(call_id, str) and isinstance(name, str):
                names[call_id] = name
    return names


def _called_tools(model_input: str | list[TResponseInputItem]) -> set[str]:
    return {name for name in _call_names(model_input).values() if not name.startswith("transfer_to_")}


def _tool_outputs(model_input: str | list[TResponseInputItem]) -> list[str]:
    if isinstance(model_input, str):
        return []
    names = _call_names(model_input)
    outputs: list[str] = []
    for item in model_input:
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        call_name = names.get(str(item.get("call_id") or ""), "")
        if call_name.startswith("transfer_to_"):
            continue
        output = item.get("output")
        if isinstance(output, str):
            outputs.append(output)
        elif isinstance(output, list):
            texts = [str(part.get("text")) for part in output if isinstance(part, dict) and part.get("text")]
            if texts:
                outputs.append("\n".join(texts))
    return outputs


def _call_id() -> str:
    global _CALL
    _CALL += 1
    return f"call_{_CALL}_{uuid4().hex[:8]}"


def _response_id() -> str:
    return f"local_{uuid4().hex[:12]}"
