from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from api.schemas.brief import BriefPayload, BriefResponse
from api.sse_queues import register, unregister
from config import settings
from db.sqlite import write_brief, write_audit
from graph.state import AgentState

router = APIRouter()


# [image-based-campaign] Build the initial graph state (shared by both endpoints).
def _build_initial_state(
    brief_id: str,
    brand: str,
    channel: str,
    persona: str,
    key_message: str,
    constraints: dict[str, Any],
    input_image_ref: str | None = None,
) -> AgentState:
    return {
        "brief_id": brief_id,
        "brand": brand,
        "channel": channel,
        "persona": persona,
        "key_message": key_message,
        "constraints": constraints,
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
        # [image-based-campaign] Image feature fields.
        "input_image_ref": input_image_ref,
        "image_description": None,
        "image_prompt": None,
        "image_path": None,
        "image_url": None,
    }


async def _run_graph(state: AgentState, q: asyncio.Queue[str], app_queues: dict) -> None:
    """Stream the LangGraph run, forwarding SSE events and gate signals."""
    from graph.builder import compiled_graph

    run_brief_id = state["brief_id"]
    pushed_count = 0
    try:
        async for event in compiled_graph.astream(state):
            for node_name, node_state in event.items():
                all_events: list[str] = node_state.get("sse_events") or []
                for msg in all_events[pushed_count:]:
                    await q.put(msg)
                pushed_count = len(all_events)

                if node_name == "compliance":
                    compliance_pass = node_state.get("compliance_pass", False)
                    revision_count = node_state.get("revision_count", 0)
                    # judge_unavailable => fail open straight to the human gate.
                    will_gate = (
                        node_state.get("judge_unavailable", False)
                        or compliance_pass
                        or (revision_count >= settings.MAX_REVISIONS)
                    )
                    # [image-based-campaign] When the image feature is on, defer the
                    # gate signal until AFTER art_director stores the image, so the
                    # review panel loads with the image already present. The human
                    # gate node emits the signal once it runs (post art_director).
                    if will_gate and not settings.IMAGE_FEATURE_ENABLED:
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


# [image-based-campaign] Register queue + spawn the background pipeline task.
def _start_pipeline(request: Request, brief_id: str, state: AgentState) -> None:
    queue: asyncio.Queue[str] = asyncio.Queue()
    request.app.state.queues[brief_id] = queue
    register(brief_id, queue)
    task = asyncio.create_task(_run_graph(state, queue, request.app.state.queues))
    request.app.state.tasks[brief_id] = task


@router.post("/brief", response_model=BriefResponse, status_code=202)
async def submit_brief(payload: BriefPayload, request: Request) -> BriefResponse:
    """Accept a text campaign brief and kick off the LangGraph pipeline."""
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
        "brand": payload.brand, "channel": payload.channel, "persona": payload.persona,
    })

    state = _build_initial_state(
        brief_id, payload.brand, payload.channel, payload.persona,
        payload.key_message, payload.constraints,
    )
    _start_pipeline(request, brief_id, state)
    return BriefResponse(brief_id=brief_id)


# [image-based-campaign] Image → campaign: accept an uploaded image and run the
# same pipeline. The vision node describes the image and grounds the copy in it.
@router.post("/brief/image", response_model=BriefResponse, status_code=202)
async def submit_brief_image(
    request: Request,
    image: UploadFile = File(...),
    brand: str = Form(...),
    channel: str = Form(...),
    persona: str = Form(...),
    key_message: str = Form(""),
    constraints: str = Form("{}"),
) -> BriefResponse:
    if not settings.IMAGE_FEATURE_ENABLED:
        raise HTTPException(status_code=403, detail="Image feature is disabled (IMAGE_FEATURE_ENABLED).")

    try:
        constraints_dict = json.loads(constraints or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="constraints must be valid JSON.")

    brief_id = str(uuid.uuid4())

    # [image-based-campaign] Persist the upload to disk; keep only the path in state.
    from services.image_store import get_image_store
    image_bytes = await image.read()
    ref = get_image_store().save(image_bytes, "uploads", brief_id, name=image.filename)

    write_brief(
        brief_id=brief_id, brand=brand, channel=channel, persona=persona,
        key_message=key_message or "(derived from uploaded image)", constraints=constraints_dict,
    )
    write_audit(brief_id, "brief_submitted", {
        "brand": brand, "channel": channel, "persona": persona, "input_image": ref.url,
    })

    state = _build_initial_state(
        brief_id, brand, channel, persona, key_message, constraints_dict,
        input_image_ref=ref.path,
    )
    _start_pipeline(request, brief_id, state)
    return BriefResponse(brief_id=brief_id)
