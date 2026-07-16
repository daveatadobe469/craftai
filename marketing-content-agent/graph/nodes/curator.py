from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import mlflow

from config import settings
from db.sqlite import update_brief_status, write_audit, write_draft
from graph.state import AgentState
from rag import chroma_client as cc
from rag import embedder


async def curator_node(state: AgentState) -> AgentState:
    """
    Stage 5 — Curator Node
    1. Use human_edits (if provided) or the original draft as final content.
    2. Normalise, chunk with RecursiveCharacterTextSplitter.
    3. Embed and upsert chunks to approved_campaigns collection.
    4. Write audit record to SQLite.
    5. Log index_latency_ms to MLflow.
    6. Post completion SSE event.
    """
    errors: list[str] = list(state.get("errors") or [])
    sse_events: list[str] = list(state.get("sse_events") or [])

    try:
        brief_id = state["brief_id"]
        brand = state["brand"]
        channel = state["channel"]
        persona = state["persona"]
        human_decision = state.get("human_decision", "approved")
        human_edits = state.get("human_edits")
        draft = state.get("draft", "")
        revision_count = state.get("revision_count") or 0
        judge_score = state.get("judge_score", 0.0)
        ragas_scores = state.get("ragas_scores") or {}
        mlflow_run_id = state.get("mlflow_run_id", "")

        final_content = human_edits.strip() if human_edits and human_edits.strip() else draft
        final_content = " ".join(final_content.split())

        sse_events.append("[Curator] Normalising and chunking approved content…")

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=32,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(final_content)
        if not chunks:
            chunks = [final_content]

        loop = asyncio.get_event_loop()
        t0 = time.monotonic()

        embeddings = await loop.run_in_executor(None, embedder.encode, chunks)

        doc_base_id = str(uuid.uuid4())
        doc_ids = [f"{doc_base_id}_{i}" for i in range(len(chunks))]
        metadatas: list[dict[str, Any]] = [
            {
                "brief_id": brief_id,
                "brand": brand,
                "channel": channel,
                "persona": persona,
                "revision_count": revision_count,
                "human_decision": human_decision or "approved",
                "judge_score": judge_score,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "doc_base_id": doc_base_id,
            }
            for i in range(len(chunks))
        ]

        await loop.run_in_executor(
            None,
            lambda: cc.upsert_documents(
                collection_name=cc.APPROVED_CAMPAIGNS,
                ids=doc_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            ),
        )

        index_latency_ms = int((time.monotonic() - t0) * 1000)

        draft_id = await loop.run_in_executor(
            None,
            lambda: write_draft(
                brief_id=brief_id,
                content=final_content,
                revision_count=revision_count,
                metadata=state.get("draft_metadata") or {},
                judge_score=judge_score,
                compliance_pass=state.get("compliance_pass", False),
                human_decision=human_decision,
                human_edits=human_edits,
                reviewed_by=state.get("reviewed_by"),
                reviewed_at=str(state.get("reviewed_at") or ""),
                ragas_scores=ragas_scores,
                mlflow_run_id=mlflow_run_id,
            ),
        )

        await loop.run_in_executor(
            None,
            write_audit,
            brief_id,
            "content_indexed",
            {
                "doc_base_id": doc_base_id,
                "chunks": len(chunks),
                "index_latency_ms": index_latency_ms,
                "draft_id": draft_id,
            },
            "curator",
        )

        await loop.run_in_executor(
            None,
            update_brief_status,
            brief_id,
            "complete",
        )

        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("craftai_campaigns")
            with mlflow.start_run(run_name=brief_id, nested=False):
                mlflow.log_metric("index_latency_ms", index_latency_ms)
                mlflow.log_metric("chunks_indexed", len(chunks))
        except Exception:
            pass

        sse_events.append(
            f"[Curator] Content indexed. {len(chunks)} chunk(s) upserted to KB. "
            f"Doc ID: {doc_base_id}. Latency: {index_latency_ms}ms."
        )
        sse_events.append("__pipeline_complete__")

        return {
            **state,
            "indexed_doc_id": doc_base_id,
            "errors": errors,
            "sse_events": sse_events,
        }

    except Exception as exc:
        errors.append(f"Curator error: {exc}")
        sse_events.append(f"[Curator] ERROR: {exc}")
        sse_events.append("__pipeline_complete__")
        return {**state, "errors": errors, "sse_events": sse_events}
