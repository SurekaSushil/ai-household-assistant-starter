from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """JSON accepted by POST /chat."""

    message: str = Field(min_length=1, max_length=10_000)
    system_prompt: str = Field(
        default=(
            "You are a careful household equipment assistant. "
            "Answer clearly, state uncertainty, and do not invent part numbers."
        ),
        max_length=2_000,
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    """JSON returned by POST /chat."""

    answer: str
    model: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None


class SearchRequest(BaseModel):
    """JSON accepted by POST /search."""

    question: str = Field(min_length=1, max_length=2_000)
    document_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class SearchHit(BaseModel):
    """One retrieved manual chunk."""

    score: float
    document_id: str
    filename: str
    page_number: int
    text: str


class RagRequest(BaseModel):
    """JSON accepted by POST /ask."""

    question: str = Field(min_length=1, max_length=2_000)
    document_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class RagSource(BaseModel):
    """Application-generated citation metadata for one retrieved source."""

    source_id: int
    filename: str
    page_number: int
    score: float
    excerpt: str


class RagResponse(BaseModel):
    """RAG answer plus inspectable sources built by the application."""

    answer: str
    sources: list[RagSource]
    model: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
