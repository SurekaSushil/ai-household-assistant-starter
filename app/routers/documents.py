from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from qdrant_client import models

from app.config import get_settings
from app.services.chunking import chunk_pages
from app.services.ollama_client import (
    OllamaResponseError,
    OllamaUnavailableError,
    embed_texts,
)
from app.services.pdf_service import PdfExtractionError, extract_pdf_pages
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    settings = get_settings()
    filename = file.filename or "document.pdf"
    content = await file.read()

    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF is too large.")

    try:
        pages = extract_pdf_pages(content)
    except PdfExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    chunks = chunk_pages(
        pages,
        chunk_size=settings.chunk_size_chars,
        overlap=settings.chunk_overlap_chars,
    )
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks were created from the PDF.")

    document_id = str(uuid4())
    file_path = Path(settings.upload_directory) / f"{document_id}.pdf"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(content)

    texts = [chunk.text for chunk in chunks]
    try:
        embeddings = await embed_texts(texts, settings)
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    store = get_vector_store()
    store.ensure_collection(len(embeddings[0]))

    points = []
    for chunk, vector in zip(chunks, embeddings, strict=True):
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            )
        )
    store.upsert_chunks(points)

    return {
        "document_id": document_id,
        "filename": filename,
        "pages_with_text": len(pages),
        "chunks_stored": len(points),
        "embedding_model": settings.ollama_embedding_model,
        "saved_path": str(file_path),
    }
