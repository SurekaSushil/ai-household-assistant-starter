from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.routers.agent import router as agent_router
from app.routers.documents import router as documents_router
from app.routers.rag import router as rag_router
from app.routers.search import router as search_router
from app.schemas import ChatRequest, ChatResponse
from app.services.ollama_client import (
    OllamaResponseError,
    OllamaUnavailableError,
    chat_with_ollama,
    stream_chat_with_ollama,
)

app = FastAPI(
    title="AI Household Assistant",
    description="A learning project that sends chat requests to a local Ollama model.",
    version="0.1.0",
)

app.include_router(documents_router)
app.include_router(search_router)
app.include_router(rag_router)
app.include_router(agent_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "AI Household Assistant is running.",
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
async def model_info() -> dict[str, str]:
    settings = get_settings()
    return {"model": settings.ollama_model}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    try:
        return await chat_with_ollama(request, settings)
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream assistant tokens as plain text as Ollama generates them."""

    settings = get_settings()

    async def token_stream() -> AsyncIterator[str]:
        # Once streaming starts, status codes are already sent, so surface
        # failures as text instead of raising HTTPException mid-stream.
        try:
            async for chunk in stream_chat_with_ollama(request, settings):
                yield chunk
        except OllamaUnavailableError as exc:
            yield f"[error] {exc}"
        except OllamaResponseError as exc:
            yield f"[error] {exc}"

    return StreamingResponse(
        token_stream(),
        media_type="text/plain; charset=utf-8",
    )
