from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel

from config import get_grader_llm, get_retrieval_llm, settings
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


def _parse_batch_scores(raw: str, expected: int) -> list[float] | None:
    """[rag-perf] Pull `expected` floats out of the grader's JSON reply.
    Returns None when the shape is wrong, so the caller can fall back."""
    match = re.search(r'\[[^\]]*\]', raw, re.DOTALL)
    if not match:
        return None
    try:
        scores = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(scores, list) or len(scores) != expected:
        return None
    try:
        return [min(1.0, max(0.0, float(s))) for s in scores]
    except (TypeError, ValueError):
        return None


async def crag_grade_batch(
    chunks: list[str], query: str, llm: BaseChatModel
) -> list[float]:
    """
    [rag-perf] Grade every retrieved chunk in ONE LLM call instead of one call
    per chunk. Identical scoring criteria to crag_grade — the only change is
    that the query and instructions are sent once rather than N times.
    Falls back to per-chunk grading if the batched reply is malformed.
    """
    if not chunks:
        return []

    numbered = "\n\n".join(
        f"[{i}] {chunk}" for i, chunk in enumerate(chunks)
    )
    prompt = (
        "Rate how relevant each numbered passage is to the query.\n"
        f"Respond with ONLY a JSON array of {len(chunks)} floats between 0.0 and 1.0, "
        "in the same order as the passages. Example: [0.9, 0.2, 0.7]\n\n"
        f"Query: {query}\n\n"
        f"Passages:\n{numbered}"
    )
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, llm.invoke, prompt)
        scores = _parse_batch_scores(response.content.strip(), len(chunks))
        if scores is not None:
            return scores
    except Exception:  # noqa: BLE001 — fall back to the per-chunk path below
        pass

    # Malformed batch reply: grade individually rather than silently dropping
    # every chunk to 0.0, which would strip the generator of all context.
    return list(await asyncio.gather(*(crag_grade(c, query, llm) for c in chunks)))


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
    llm: BaseChatModel | None = None,
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
    # [rag-perf] Two different models on purpose: HyDE (write a snippet) runs on
    # the cheap retrieval model, but batched CRAG grading needs a capable one —
    # the small model silently drops relevant chunks when grading a batch. A
    # caller-supplied llm overrides only HyDE; grading always uses the grader.
    hyde_llm = llm or get_retrieval_llm()
    grader_llm = get_grader_llm()

    hypo_doc = await hyde_rewrite(brief_text, hyde_llm)

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

    # [rag-perf] One grading call for the whole result set, not one per chunk.
    grades = await crag_grade_batch(docs, brief_text, grader_llm)

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
