import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import Settings
from app.schemas import ChatRequest, ChatResponse


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""


class OllamaResponseError(RuntimeError):
    """Raised when Ollama returns an unexpected response."""


def _chat_payload(
    request: ChatRequest,
    settings: Settings,
    *,
    stream: bool,
) -> dict[str, Any]:
    return {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.message},
        ],
        "stream": stream,
        "options": {"temperature": request.temperature},
    }


async def _post_chat(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_seconds
        ) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
    except httpx.ConnectError as exc:
        raise OllamaUnavailableError(
            "Could not connect to Ollama. Start Ollama and run the model first."
        ) from exc
    except httpx.TimeoutException as exc:
        raise OllamaUnavailableError(
            "Ollama did not answer before the request timeout."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise OllamaResponseError(
            f"Ollama returned HTTP {exc.response.status_code}: "
            f"{exc.response.text[:500]}"
        ) from exc

    return response.json()


async def chat_with_ollama(
    request: ChatRequest,
    settings: Settings,
) -> ChatResponse:
    """Send one chat request to the local Ollama HTTP API."""

    payload = _chat_payload(request, settings, stream=False)
    data = await _post_chat(payload, settings)
    content = data.get("message", {}).get("content")
    if not isinstance(content, str):
        raise OllamaResponseError("Ollama response did not contain message.content.")

    return ChatResponse(
        answer=content,
        model=str(data.get("model", settings.ollama_model)),
        prompt_tokens=data.get("prompt_eval_count"),
        output_tokens=data.get("eval_count"),
    )


async def chat_messages(
    messages: list[dict[str, Any]],
    settings: Settings,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """POST a message list (and optional tools) to Ollama /api/chat."""

    payload: dict[str, Any] = {
        "model": model or settings.ollama_agent_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if tools:
        payload["tools"] = tools

    data = await _post_chat(payload, settings)
    if not isinstance(data.get("message"), dict):
        raise OllamaResponseError("Ollama response did not contain message.")
    return data


async def stream_chat_with_ollama(
    request: ChatRequest,
    settings: Settings,
) -> AsyncIterator[str]:
    """Yield assistant text chunks as Ollama streams NDJSON responses."""

    payload = _chat_payload(request, settings, stream=True)

    try:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_seconds
        ) as client:
            async with client.stream(
                "POST",
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    body = (await response.aread())[:500]
                    raise OllamaResponseError(
                        f"Ollama returned HTTP {exc.response.status_code}: "
                        f"{body.decode(errors='replace')}"
                    ) from exc

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaResponseError(
                            "Ollama stream returned invalid JSON."
                        ) from exc

                    content = data.get("message", {}).get("content")
                    if isinstance(content, str) and content:
                        yield content
    except httpx.ConnectError as exc:
        raise OllamaUnavailableError(
            "Could not connect to Ollama. Start Ollama and run the model first."
        ) from exc
    except httpx.TimeoutException as exc:
        raise OllamaUnavailableError(
            "Ollama did not answer before the request timeout."
        ) from exc


async def embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    """Turn text strings into embedding vectors via Ollama /api/embed."""

    try:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_seconds
        ) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/embed",
                json={
                    "model": settings.ollama_embedding_model,
                    "input": texts,
                },
            )
            response.raise_for_status()
    except httpx.ConnectError as exc:
        raise OllamaUnavailableError(
            "Could not connect to Ollama. Start Ollama and run the model first."
        ) from exc
    except httpx.TimeoutException as exc:
        raise OllamaUnavailableError(
            "Ollama did not answer before the request timeout."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise OllamaResponseError(
            f"Ollama returned HTTP {exc.response.status_code}: "
            f"{exc.response.text[:500]}"
        ) from exc

    data = response.json()
    embeddings = data.get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        raise OllamaResponseError(
            "Ollama returned an unexpected embedding response."
        )
    return embeddings
