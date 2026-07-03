from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from api.schemas.decision import SearchRequest, SearchResponse, SearchResult
from rag import chroma_client as cc
from rag import embedder

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def semantic_search(payload: SearchRequest) -> SearchResponse:
    """
    Semantic similarity search against one of the three ChromaDB collections.
    """
    valid_collections = {cc.APPROVED_CAMPAIGNS, cc.SOCIAL_CONTENT, cc.BRAND_GUIDELINES}
    if payload.collection not in valid_collections:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid collection. Must be one of: {', '.join(sorted(valid_collections))}",
        )

    loop = asyncio.get_event_loop()

    count = await loop.run_in_executor(
        None, cc.collection_count, payload.collection
    )
    if count == 0:
        return SearchResponse(
            query=payload.query,
            collection=payload.collection,
            results=[],
            total=0,
        )

    query_embedding = await loop.run_in_executor(
        None, embedder.encode_single, payload.query
    )

    where = payload.filters if payload.filters else None
    result = await loop.run_in_executor(
        None,
        lambda: cc.query_collection(
            collection_name=payload.collection,
            query_embeddings=[query_embedding],
            n_results=min(payload.top_k, count),
            where=where,
        ),
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    results = [
        SearchResult(
            document=doc,
            metadata=meta or {},
            score=round(max(0.0, 1.0 - dist), 4),
        )
        for doc, meta, dist in zip(docs, metas, dists)
    ]

    return SearchResponse(
        query=payload.query,
        collection=payload.collection,
        results=results,
        total=len(results),
    )
