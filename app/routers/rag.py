from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.schemas import RagRequest, RagResponse
from app.services.ollama_client import OllamaResponseError, OllamaUnavailableError
from app.services.rag_service import answer_with_rag

router = APIRouter(tags=["rag"])


@router.post("/ask", response_model=RagResponse)
async def ask_manual(request: RagRequest) -> RagResponse:
    settings = get_settings()
    try:
        return await answer_with_rag(
            request.question,
            settings,
            document_id=request.document_id,
            top_k=request.top_k,
            temperature=request.temperature,
        )
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
