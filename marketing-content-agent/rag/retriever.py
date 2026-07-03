from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel

from config import settings
from rag import chroma_client, embedder


async def hyde_rewrite(brief_text: str, llm: BaseChatModel) -> str:
    """
    Hypothetical Document Embeddings (HyDE): ask the LLM to write an ideal
    marketing snippet that answers the brief, then embed that snippet.
    Returns the hypothetical document text (the caller embeds it).
    """
    prompt = (
        "You are a senior marketing copywriter. "
        "Write a concise, high-quality marketing snippet that perfectly addresses "
        "the following campaign brief. Output ONLY the marketing copy, no preamble.\n\n"
        f"Brief:\n{brief_text}"
    )
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, llm.invoke, prompt)
    return response.content.strip()


async def crag_grade(chunk: str, query: str, llm: BaseChatModel) -> float:
    """
    CRAG self-grader: ask the LLM to score the relevance of `chunk` to `query`.
    Returns a float in [0.0, 1.0]. Defaults to 0.0 on any parse failure.
    """
    prompt = (
        'Rate how relevant the following retrieved passage is to the given query.\n'
        'Respond with a JSON object only: {"score": <float between 0.0 and 1.0>}\n\n'
        f'Query: {query}\n\n'
        f'Passage:\n{chunk}'
    )
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, llm.invoke, prompt)
        raw = response.content.strip()
        match = re.search(r'\{.*?"score"\s*:\s*([0-9.]+).*?\}', raw, re.DOTALL)
        if match:
            return float(match.group(1))
        data = json.loads(raw)
        return float(data.get("score", 0.0))
    except (json.JSONDecodeError, ValueError, AttributeError):
        return 0.0


async def retrieve(
    query: str,
    collection_name: str,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Embed `query` and run a nearest-neighbour search in the given collection.
    Returns a list of dicts with keys: document, metadata, distance, score.
    """
    loop = asyncio.get_event_loop()

    query_embedding = await loop.run_in_executor(None, embedder.encode_single, query)

    count = await loop.run_in_executor(
        None, chroma_client.collection_count, collection_name
    )
    if count == 0:
        return []

    result = await loop.run_in_executor(
        None,
        lambda: chroma_client.query_collection(
            collection_name=collection_name,
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where=where,
        ),
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    chunks: list[dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        cosine_sim = 1.0 - dist
        chunks.append({
            "document": doc,
            "metadata": meta or {},
            "distance": dist,
            "score": max(0.0, cosine_sim),
        })
    return chunks


async def retrieve_with_hyde(
    brief_text: str,
    collection_name: str,
    llm: BaseChatModel,
    top_k: int = 5,
    crag_threshold: float | None = None,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Full HyDE + CRAG retrieval pipeline:
    1. Generate hypothetical document via LLM.
    2. Retrieve top_k chunks using the hypothetical doc embedding.
    3. Self-grade each chunk with CRAG; discard below threshold.
    Returns surviving chunks sorted by score descending.
    """
    threshold = crag_threshold if crag_threshold is not None else settings.CRAG_THRESHOLD

    hypo_doc = await hyde_rewrite(brief_text, llm)

    loop = asyncio.get_event_loop()
    count = await loop.run_in_executor(
        None, chroma_client.collection_count, collection_name
    )
    if count == 0:
        return []

    hypo_embedding = await loop.run_in_executor(
        None, embedder.encode_single, hypo_doc
    )

    result = await loop.run_in_executor(
        None,
        lambda: chroma_client.query_collection(
            collection_name=collection_name,
            query_embeddings=[hypo_embedding],
            n_results=min(top_k, count),
            where=where,
        ),
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    grade_tasks = [crag_grade(doc, brief_text, llm) for doc in docs]
    grades = await asyncio.gather(*grade_tasks)

    chunks: list[dict[str, Any]] = []
    for doc, meta, dist, grade in zip(docs, metas, dists, grades):
        if grade >= threshold:
            chunks.append({
                "document": doc,
                "metadata": meta or {},
                "distance": dist,
                "score": grade,
            })

    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks
