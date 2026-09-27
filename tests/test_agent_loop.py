import asyncio
from typing import Any

from app.agent.loop import MAX_STEPS_MESSAGE, run_agent_chat
from app.config import Settings
from app.schemas import AgentChatRequest


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def _settings(**overrides: Any) -> Settings:
    values = {
        "ollama_agent_model": "mock-agent",
        "agent_max_steps": 5,
        "request_timeout_seconds": 5.0,
    }
    values.update(overrides)
    return Settings(**values)


def test_calculator_round_trip_with_mocked_ollama(monkeypatch: Any) -> None:
    calls: list[list[dict[str, Any]]] = []

    async def fake_chat_messages(
        messages: list[dict[str, Any]],
        settings: Settings,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append(messages)
        if len(calls) == 1:
            return {
                "model": "mock-agent",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "calculator",
                                "arguments": {"expression": "(37 * 19) + 8"},
                            }
                        }
                    ],
                },
                "prompt_eval_count": 11,
                "eval_count": 4,
            }
        return {
            "model": "mock-agent",
            "message": {
                "role": "assistant",
                "content": "The result is 711.",
            },
            "prompt_eval_count": 20,
            "eval_count": 6,
        }

    monkeypatch.setattr("app.agent.loop.chat_messages", fake_chat_messages)
    response = _run(
        run_agent_chat(
            AgentChatRequest(message="What is (37 * 19) + 8?"),
            _settings(),
        )
    )
    assert response.answer == "The result is 711."
    assert response.steps == 2
    assert response.prompt_tokens == 31
    assert response.output_tokens == 10
    assert len(response.tools_used) == 1
    record = response.tools_used[0]
    assert record.name == "calculator"
    assert record.arguments == {"expression": "(37 * 19) + 8"}
    assert record.success is True
    assert '"result":711.0' in record.result_preview.replace(" ", "")

    second_turn = calls[1]
    assert second_turn[-1]["role"] == "tool"
    assert second_turn[-1]["tool_name"] == "calculator"
    assert "711" in second_turn[-1]["content"]


def test_unknown_tool_does_not_crash_loop(monkeypatch: Any) -> None:
    calls = {"n": 0}

    async def fake_chat_messages(
        messages: list[dict[str, Any]],
        settings: Settings,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "model": "mock-agent",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "not_a_real_tool", "arguments": {}}}
                    ],
                },
            }
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert tool_messages
        assert "Unknown tool" in tool_messages[-1]["content"]
        return {
            "model": "mock-agent",
            "message": {
                "role": "assistant",
                "content": "I could not use that tool.",
            },
        }

    monkeypatch.setattr("app.agent.loop.chat_messages", fake_chat_messages)
    response = _run(
        run_agent_chat(AgentChatRequest(message="ignore this"), _settings())
    )
    assert response.answer == "I could not use that tool."
    assert response.tools_used[0].success is False
    assert response.tools_used[0].name == "not_a_real_tool"


def test_max_steps_stops_without_live_ollama(monkeypatch: Any) -> None:
    async def fake_chat_messages(
        messages: list[dict[str, Any]],
        settings: Settings,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "model": "mock-agent",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculator",
                            "arguments": {"expression": "1 + 1"},
                        }
                    }
                ],
            },
        }

    monkeypatch.setattr("app.agent.loop.chat_messages", fake_chat_messages)
    response = _run(
        run_agent_chat(
            AgentChatRequest(message="keep calling tools"),
            _settings(agent_max_steps=2),
        )
    )
    assert response.answer == MAX_STEPS_MESSAGE
    assert response.steps == 2
    assert len(response.tools_used) == 2


def test_search_manual_injects_request_document_id(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_search(
        question: str,
        settings: Settings,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        captured["document_id"] = document_id
        captured["question"] = question
        return [
            {
                "score": 0.5,
                "document_id": document_id,
                "filename": "manual.pdf",
                "page_number": 3,
                "text": "See the drain filter.",
            }
        ]

    async def fake_chat_messages(
        messages: list[dict[str, Any]],
        settings: Settings,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not any(m.get("role") == "tool" for m in messages):
            return {
                "model": "mock-agent",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_manual",
                                "arguments": {"query": "drain filter"},
                            }
                        }
                    ],
                },
            }
        return {
            "model": "mock-agent",
            "message": {
                "role": "assistant",
                "content": "Clean the drain filter (manual.pdf, page 3).",
            },
        }

    monkeypatch.setattr("app.agent.tools.search_manual_chunks", fake_search)
    monkeypatch.setattr("app.agent.loop.chat_messages", fake_chat_messages)
    response = _run(
        run_agent_chat(
            AgentChatRequest(
                message="How do I clean the drain filter?",
                document_id="doc-42",
            ),
            _settings(),
        )
    )
    assert captured["document_id"] == "doc-42"
    assert captured["question"] == "drain filter"
    assert response.tools_used[0].success is True
    assert response.tools_used[0].arguments["document_id"] == "doc-42"


def test_no_tool_call_returns_first_answer(monkeypatch: Any) -> None:
    async def fake_chat_messages(
        messages: list[dict[str, Any]],
        settings: Settings,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "model": "mock-agent",
            "message": {"role": "assistant", "content": "Hello."},
        }

    monkeypatch.setattr("app.agent.loop.chat_messages", fake_chat_messages)
    response = _run(
        run_agent_chat(AgentChatRequest(message="hi"), _settings())
    )
    assert response.answer == "Hello."
    assert response.steps == 1
    assert response.tools_used == []
