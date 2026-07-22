from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from api.schemas.brief import BriefPayload, BriefResponse
from api.sse_queues import register, unregister
from config import settings
from db.sqlite import write_brief, write_audit
from graph.state import AgentState

router = APIRouter()


@router.post("/brief", response_model=BriefResponse, status_code=202)
async def submit_brief(payload: BriefPayload, request: Request) -> BriefResponse:
    """
    Accept a campaign brief, persist it, and kick off the LangGraph pipeline.
    Returns immediately with brief_id; client polls /status or streams /stream.
    """
    brief_id = str(uuid.uuid4())

    write_brief(
        brief_id=brief_id,
        brand=payload.brand,
        channel=payload.channel,
        persona=payload.persona,
        key_message=payload.key_message,
        constraints=payload.constraints,
    )

    write_audit(brief_id, "brief_submitted", {
        "brand": payload.brand,
        "channel": payload.channel,
        "persona": payload.persona,
    })

    queue: asyncio.Queue[str] = asyncio.Queue()
    request.app.state.queues[brief_id] = queue
    register(brief_id, queue)

    initial_state: AgentState = {
        "brief_id": brief_id,
        "brand": payload.brand,
        "channel": payload.channel,
        "persona": payload.persona,
        "key_message": payload.key_message,
        "constraints": payload.constraints,
        "retrieved_campaigns": [],
        "retrieved_social": [],
        "retrieved_guidelines": [],
        "draft": "",
        "draft_metadata": {},
        "revision_count": 0,
        "rule_violations": [],
        "judge_score": 0.0,
        "judge_evidence": "",
        "compliance_pass": False,
        "human_decision": None,
        "human_edits": None,
        "reviewed_by": None,
        "reviewed_at": None,
        "indexed_doc_id": None,
        "ragas_scores": {},
        "mlflow_run_id": "",
        "plan": [],
        "errors": [],
        "sse_events": [],
    }

    from graph.builder import compiled_graph

    app_queues = request.app.state.queues

    async def _run_graph(state: AgentState, q: asyncio.Queue[str]) -> None:
        run_brief_id = state["brief_id"]
        pushed_count = 0
        try:
            async for event in compiled_graph.astream(state):
                for node_name, node_state in event.items():
                    all_events: list[str] = node_state.get("sse_events") or []

                    new_events = all_events[pushed_count:]
                    for msg in new_events:
                        await q.put(msg)
                    pushed_count = len(all_events)

                    if node_name == "compliance":
                        compliance_pass = node_state.get("compliance_pass", False)
                        revision_count = node_state.get("revision_count", 0)
                        will_gate = compliance_pass or (revision_count >= settings.MAX_REVISIONS)
                        if will_gate:
                            await q.put("__human_action_required__")

                    if node_name == "human_gate":
                        if node_state.get("human_decision") == "rejected":
                            await q.put("__pipeline_complete__")

                    if node_name == "curator":
                        await q.put("__pipeline_complete__")

            await q.put("__done__")
        except Exception as exc:
            await q.put(f"__error__:{exc}")
            await q.put("__done__")
        finally:
            unregister(run_brief_id)
            app_queues.pop(run_brief_id, None)

    task = asyncio.create_task(_run_graph(initial_state, queue))
    request.app.state.tasks[brief_id] = task

    return BriefResponse(brief_id=brief_id)
