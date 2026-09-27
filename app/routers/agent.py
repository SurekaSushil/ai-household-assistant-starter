from fastapi import APIRouter, HTTPException

from app.agent.loop import run_agent_chat
from app.config import get_settings
from app.schemas import AgentChatRequest, AgentChatResponse
from app.services.ollama_client import OllamaResponseError, OllamaUnavailableError

router = APIRouter(tags=["agent"])


@router.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    settings = get_settings()
    try:
        return await run_agent_chat(request, settings)
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
