from __future__ import annotations

import threading
from typing import Any

import chromadb
from chromadb import Collection
from chromadb.config import Settings as ChromaSettings

from config import settings

# Collection name constants
BRAND_GUIDELINES: str = "brand_guidelines"
APPROVED_CAMPAIGNS: str = "approved_campaigns"
SOCIAL_CONTENT: str = "social_content"

_lock = threading.Lock()
_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    """Return the singleton PersistentClient, creating it on first call."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = chromadb.PersistentClient(
                    path=settings.CHROMA_PERSIST_DIR,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
    return _client


def get_collection(name: str) -> Collection:
    """Get or create a collection with cosine distance metric."""
    client = get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def init_collections() -> None:
    """Ensure all three collections exist."""
    for name in (BRAND_GUIDELINES, APPROVED_CAMPAIGNS, SOCIAL_CONTENT):
        get_collection(name)


def upsert_documents(
    collection_name: str,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    """Upsert a batch of documents into the named collection."""
    col = get_collection(collection_name)
    col.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query_collection(
    collection_name: str,
    query_embeddings: list[list[float]],
    n_results: int = 5,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Query a collection and return the raw ChromaDB result dict."""
    col = get_collection(collection_name)
    kwargs: dict[str, Any] = {
        "query_embeddings": query_embeddings,
        "n_results": min(n_results, col.count() or 1),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return col.query(**kwargs)


def collection_count(collection_name: str) -> int:
    return get_collection(collection_name).count()
