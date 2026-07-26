from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

# Uvicorn's Windows reloader can fail to find pywin32 DLLs unless we add them
# before portalocker/Qdrant import and lock the local storage folder.
_pywin32_dir = Path(sys.prefix) / "Lib" / "site-packages" / "pywin32_system32"
if _pywin32_dir.is_dir():
    os.environ["PATH"] = str(_pywin32_dir) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(_pywin32_dir))

import pywintypes  # noqa: E402,F401
from qdrant_client import QdrantClient, models  # noqa: E402

from app.config import Settings


class VectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = QdrantClient(path=settings.qdrant_path)

    def ensure_collection(self, vector_size: int) -> None:
        name = self.settings.qdrant_collection
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert_chunks(self, points: list[models.PointStruct]) -> None:
        self.client.upsert(
            collection_name=self.settings.qdrant_collection,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        document_id: str | None = None,
    ):
        query_filter = None
        if document_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        return self.client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points


@lru_cache
def get_vector_store() -> VectorStore:
    """Reuse one local Qdrant client; opening multiple path-based clients conflicts."""

    from app.config import get_settings

    return VectorStore(get_settings())
