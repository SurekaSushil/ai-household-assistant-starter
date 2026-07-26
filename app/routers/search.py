from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.schemas import SearchHit, SearchRequest
from app.services.ollama_client import OllamaResponseError, OllamaUnavailableError
from app.services.search_service import search_manual_chunks

router = APIRouter(tags=["search"])


@router.post("/search", response_model=list[SearchHit])
async def search_documents(request: SearchRequest) -> list[dict]:
    settings = get_settings()
    try:
        return await search_manual_chunks(
            request.question,
            settings,
            top_k=request.top_k,
            document_id=request.document_id,
        )
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
