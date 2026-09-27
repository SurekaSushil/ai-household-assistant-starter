import json
from typing import Any

from app.agent.tools import (
    ToolDefinition,
    execute_approved_tool,
    get_tool_registry,
    is_tool_error_payload,
    ollama_tool_schemas,
)
from app.config import Settings
from app.schemas import AgentChatRequest, AgentChatResponse, ToolCallRecord
from app.services.ollama_client import chat_messages

RESULT_PREVIEW_CHARS = 240
MAX_STEPS_MESSAGE = (
    "Stopped: the agent reached AGENT_MAX_STEPS without a final answer."
)


async def run_agent_chat(
    request: AgentChatRequest,
    settings: Settings,
    *,
    registry: dict[str, ToolDefinition] | None = None,
) -> AgentChatResponse:
    """Ask the model, execute approved tool calls, and repeat until an answer or the step cap."""

    tools = registry if registry is not None else get_tool_registry()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": request.system_prompt},
        {"role": "user", "content": request.message},
    ]
    tools_used: list[ToolCallRecord] = []
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    model_name = settings.ollama_agent_model
    steps = 0

    for step in range(1, settings.agent_max_steps + 1):
        steps = step
        data = await chat_messages(
            messages,
            settings,
            model=settings.ollama_agent_model,
            temperature=request.temperature,
            tools=ollama_tool_schemas(tools),
        )
        model_name = str(data.get("model", model_name))
        prompt_tokens = _add_count(prompt_tokens, data.get("prompt_eval_count"))
        output_tokens = _add_count(output_tokens, data.get("eval_count"))

        message = data.get("message") or {}
        content = message.get("content")
        content_text = content if isinstance(content, str) else ""
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []

        if not tool_calls:
            return AgentChatResponse(
                answer=content_text,
                tools_used=tools_used,
                steps=steps,
                model=model_name,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content_text,
            "tool_calls": tool_calls,
        }
        messages.append(assistant_message)

        for call in tool_calls:
            call_dict = call if isinstance(call, dict) else {}
            name, arguments = _parse_tool_call(call_dict)
            if name == "search_manual" and request.document_id:
                arguments["document_id"] = request.document_id

            result_json = await execute_approved_tool(
                name,
                arguments,
                settings=settings,
                registry=tools,
            )
            recorded_arguments = _validated_or_raw(tools, name, arguments)
            tools_used.append(
                ToolCallRecord(
                    name=name or "unknown",
                    arguments=recorded_arguments,
                    result_preview=_preview(result_json),
                    success=not is_tool_error_payload(result_json),
                )
            )
            messages.append(_tool_result_message(call_dict, name, result_json))

    return AgentChatResponse(
        answer=MAX_STEPS_MESSAGE,
        tools_used=tools_used,
        steps=steps,
        model=model_name,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )


def _parse_tool_call(call: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(call, dict):
        return "", {}
    function = call.get("function") or {}
    if not isinstance(function, dict):
        return "", {}
    name = function.get("name")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return str(name or ""), dict(arguments)


def _validated_or_raw(
    registry: dict[str, ToolDefinition],
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    tool = registry.get(name)
    if tool is None:
        return arguments
    try:
        return tool.args_model.model_validate(arguments).model_dump()
    except Exception:
        return arguments


def _tool_result_message(
    call: dict[str, Any],
    name: str,
    result_json: str,
) -> dict[str, Any]:
    """Ollama native API keys tool results by tool_name; keep tool_call_id if present."""

    message: dict[str, Any] = {
        "role": "tool",
        "content": result_json,
        "tool_name": name,
    }
    call_id = call.get("id") or call.get("tool_call_id")
    if call_id:
        message["tool_call_id"] = call_id
    return message


def _preview(text: str, limit: int = RESULT_PREVIEW_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _add_count(current: int | None, incoming: Any) -> int | None:
    if not isinstance(incoming, int):
        return current
    return (current or 0) + incoming
