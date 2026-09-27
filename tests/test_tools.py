import asyncio
import json
from typing import Any

from pydantic import BaseModel, Field

from app.agent.tools import (
    CalculatorArgs,
    ToolDefinition,
    execute_approved_tool,
    is_tool_error_payload,
)
from app.config import Settings


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def test_calculator_success_json() -> None:
    result = _run(
        execute_approved_tool("calculator", {"expression": "(37 * 19) + 8"})
    )
    assert json.loads(result) == {"result": 711.0}


def test_calculator_power_json() -> None:
    result = _run(execute_approved_tool("calculator", {"expression": "2 ** 10"}))
    assert json.loads(result) == {"result": 1024.0}


def test_unknown_tool_returns_error_json() -> None:
    result = _run(execute_approved_tool("os.system", {"command": "dir"}))
    payload = json.loads(result)
    assert "error" in payload
    assert "Unknown tool" in payload["error"]


def test_division_by_zero_returns_error_json() -> None:
    result = _run(execute_approved_tool("calculator", {"expression": "1 / 0"}))
    payload = json.loads(result)
    assert "error" in payload
    assert is_tool_error_payload(result)


def test_import_payload_is_rejected() -> None:
    result = _run(
        execute_approved_tool(
            "calculator",
            {"expression": '__import__("os").system("dir")'},
        )
    )
    payload = json.loads(result)
    assert "error" in payload
    assert "Unsupported" in payload["error"]


def test_invalid_arguments_never_reach_handler() -> None:
    called: list[Any] = []

    def handler(args: CalculatorArgs) -> dict[str, str]:
        called.append(args)
        return {"ok": "yes"}

    registry = {
        "probe": ToolDefinition(
            name="probe",
            description="test",
            args_model=CalculatorArgs,
            handler=handler,
        )
    }
    result = _run(
        execute_approved_tool(
            "probe",
            {"expression": ""},
            registry=registry,
        )
    )
    assert called == []
    assert "error" in json.loads(result)


def test_search_manual_wraps_search_service(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_search(
        question: str,
        settings: Settings,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        captured["question"] = question
        captured["top_k"] = top_k
        captured["document_id"] = document_id
        return [
            {
                "score": 0.91,
                "document_id": "doc-1",
                "filename": "washer.pdf",
                "page_number": 12,
                "text": "Clean the filter monthly.",
            }
        ]

    monkeypatch.setattr("app.agent.tools.search_manual_chunks", fake_search)
    result = _run(
        execute_approved_tool(
            "search_manual",
            {"query": "clean filter", "document_id": "doc-1", "top_k": 3},
            settings=Settings(),
        )
    )
    payload = json.loads(result)
    assert captured == {
        "question": "clean filter",
        "top_k": 3,
        "document_id": "doc-1",
    }
    assert payload == [
        {
            "page": 12,
            "score": 0.91,
            "filename": "washer.pdf",
            "text": "Clean the filter monthly.",
        }
    ]


class _TinyArgs(BaseModel):
    n: int = Field(ge=1, le=2)


def test_validation_error_is_json() -> None:
    def handler(args: _TinyArgs) -> dict[str, int]:
        return {"n": args.n}

    registry = {
        "tiny": ToolDefinition(
            name="tiny",
            description="test",
            args_model=_TinyArgs,
            handler=handler,
        )
    }
    result = _run(execute_approved_tool("tiny", {"n": 99}, registry=registry))
    assert "error" in json.loads(result)
