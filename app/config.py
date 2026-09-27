from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from defaults and an optional .env file."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:1b"
    request_timeout_seconds: float = 180.0

    upload_directory: str = "data/uploads"
    max_upload_mb: int = 20
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200
    # Skip French/Spanish (etc.) pages when ingesting bilingual manuals.
    skip_non_english_pages: bool = True
    ollama_embedding_model: str = "embeddinggemma"
    qdrant_path: str = "data/qdrant"
    qdrant_collection: str = "manual_chunks"
    rag_top_k: int = 5
    rag_min_score: float = 0.25

    ollama_agent_model: str = "qwen3:4b-instruct"
    agent_max_steps: int = 5
    # Reserved for session 3.6 (SQLite memory). Unused by the agent loop today.
    memory_database: str = "data/memory.db"
    memory_messages: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Create settings once and reuse them for later requests."""

    return Settings()
