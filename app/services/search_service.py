from app.config import Settings
from app.services.ollama_client import embed_texts
from app.services.vector_store import get_vector_store


async def search_manual_chunks(
    question: str,
    settings: Settings,
    *,
    top_k: int | None = None,
    document_id: str | None = None,
) -> list[dict]:
    """Embed a question and return the nearest stored manual chunks as JSON."""

    store = get_vector_store()
    query_vector = (await embed_texts([question], settings))[0]
    hits = store.search(
        query_vector,
        top_k=top_k or settings.rag_top_k,
        document_id=document_id,
    )

    return [
        {
            "score": hit.score,
            "document_id": hit.payload["document_id"],
            "filename": hit.payload["filename"],
            "page_number": hit.payload["page_number"],
            "text": hit.payload["text"],
        }
        for hit in hits
    ]
