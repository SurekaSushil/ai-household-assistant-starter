from app.config import Settings
from app.schemas import ChatRequest, RagResponse, RagSource
from app.services.ollama_client import chat_with_ollama
from app.services.search_service import search_manual_chunks

RAG_SYSTEM_PROMPT = """You answer questions about uploaded manuals.
Use only the supplied sources.
Cite supporting claims as [SOURCE n].
If the sources do not support an answer, say that the manual evidence is insufficient.
Never invent model numbers, part numbers, warnings, or procedures.
"""


def build_context_and_sources(hits: list[dict]) -> tuple[str, list[RagSource]]:
    """Build the LLM context and the application-owned sources array from the same hit order."""

    context_blocks: list[str] = []
    sources: list[RagSource] = []

    for index, hit in enumerate(hits, start=1):
        context_blocks.append(
            f"[SOURCE {index} | {hit['filename']} | page {hit['page_number']}]\n"
            f"{hit['text']}"
        )
        excerpt = hit["text"][:240].rstrip()
        if len(hit["text"]) > 240:
            excerpt += "..."
        sources.append(
            RagSource(
                source_id=index,
                filename=str(hit["filename"]),
                page_number=int(hit["page_number"]),
                score=float(hit["score"]),
                excerpt=excerpt,
            )
        )

    context = "\n\n".join(context_blocks)
    return context, sources


async def answer_with_rag(
    question: str,
    settings: Settings,
    *,
    document_id: str | None = None,
    top_k: int | None = None,
    temperature: float = 0.2,
) -> RagResponse:
    """Retrieve manual chunks, ask the chat model, and return answer + sources."""

    hits = await search_manual_chunks(
        question,
        settings,
        top_k=top_k,
        document_id=document_id,
    )
    hits = [hit for hit in hits if hit["score"] >= settings.rag_min_score]

    if not hits:
        return RagResponse(
            answer=(
                "The manual evidence is insufficient. "
                "No sufficiently relevant passages were retrieved for this question."
            ),
            sources=[],
            model=settings.ollama_model,
        )

    context, sources = build_context_and_sources(hits)
    user_prompt = f"Question: {question}\n\nSources:\n{context}"

    chat_response = await chat_with_ollama(
        ChatRequest(
            message=user_prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
            temperature=temperature,
        ),
        settings,
    )

    return RagResponse(
        answer=chat_response.answer,
        sources=sources,
        model=chat_response.model,
        prompt_tokens=chat_response.prompt_tokens,
        output_tokens=chat_response.output_tokens,
    )
