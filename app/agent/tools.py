import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.agent.calculator import calculate
from app.config import Settings
from app.services.search_service import search_manual_chunks

_COMPACT_JSON = (",", ":")


class CalculatorArgs(BaseModel):
    expression: str = Field(
        min_length=1,
        max_length=200,
        description="Arithmetic expression such as (37 * 19) + 8",
    )


class SearchManualArgs(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[..., Any]


def run_calculator(args: CalculatorArgs) -> dict[str, float]:
    return {"result": calculate(args.expression)}


async def run_search_manual(
    args: SearchManualArgs,
    settings: Settings,
) -> list[dict[str, Any]]:
    hits = await search_manual_chunks(
        args.query,
        settings,
        top_k=args.top_k,
        document_id=args.document_id,
    )
    return [
        {
            "page": hit["page_number"],
            "score": hit["score"],
            "filename": hit["filename"],
            "text": hit["text"],
        }
        for hit in hits
    ]


CALCULATOR_TOOL = ToolDefinition(
    name="calculator",
    description="Evaluate basic arithmetic.",
    args_model=CalculatorArgs,
    handler=run_calculator,
)

SEARCH_MANUAL_TOOL = ToolDefinition(
    name="search_manual",
    description=(
        "Search uploaded appliance manuals for evidence. "
        "Use before answering manual-specific questions."
    ),
    args_model=SearchManualArgs,
    handler=run_search_manual,
)

DEFAULT_TOOL_REGISTRY: dict[str, ToolDefinition] = {
    CALCULATOR_TOOL.name: CALCULATOR_TOOL,
    SEARCH_MANUAL_TOOL.name: SEARCH_MANUAL_TOOL,
}


def get_tool_registry() -> dict[str, ToolDefinition]:
    return dict(DEFAULT_TOOL_REGISTRY)


def ollama_tool_schemas(
    registry: dict[str, ToolDefinition] | None = None,
) -> list[dict[str, Any]]:
    """Convert registered Pydantic argument models into Ollama tool schemas."""

    tools = registry if registry is not None else DEFAULT_TOOL_REGISTRY
    payloads: list[dict[str, Any]] = []
    for tool in tools.values():
        schema = tool.args_model.model_json_schema()
        parameters: dict[str, Any] = {
            "type": schema.get("type", "object"),
            "properties": schema.get("properties", {}),
        }
        if "required" in schema:
            parameters["required"] = schema["required"]
        payloads.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            }
        )
    return payloads


def tool_error_json(message: str) -> str:
    return json.dumps({"error": message}, separators=_COMPACT_JSON)


def is_tool_error_payload(result_json: str) -> bool:
    try:
        payload = json.loads(result_json)
    except json.JSONDecodeError:
        return True
    return isinstance(payload, dict) and "error" in payload


async def execute_approved_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    settings: Settings | None = None,
    registry: dict[str, ToolDefinition] | None = None,
) -> str:
    """Validate arguments, then run a registered handler. Never raises to the caller."""

    tools = registry if registry is not None else DEFAULT_TOOL_REGISTRY
    tool = tools.get(name)
    if tool is None:
        return tool_error_json(f"Unknown tool: {name}")

    try:
        validated = tool.args_model.model_validate(arguments)
        result = await _call_handler(tool, validated, settings)
        return json.dumps(result, separators=_COMPACT_JSON)
    except ValidationError as exc:
        return tool_error_json(f"Invalid arguments: {exc}")
    except Exception as exc:
        return tool_error_json(str(exc))


async def _call_handler(
    tool: ToolDefinition,
    args: BaseModel,
    settings: Settings | None,
) -> Any:
    kwargs: dict[str, Any] = {}
    parameters = inspect.signature(tool.handler).parameters
    if "settings" in parameters:
        if settings is None:
            raise ValueError("This tool requires application settings.")
        kwargs["settings"] = settings
    result = tool.handler(args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result
