from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from api.schemas.decision import DecisionPayload
from api.sse_queues import push_sync
from db.sqlite import get_brief, update_brief_status, update_latest_draft_decision, write_audit
from graph.nodes.human_gate import set_decision

router = APIRouter()


class DecisionResponse(BaseModel):
    brief_id: str
    decision: str
    message: str


@router.post("/decision/{brief_id}", response_model=DecisionResponse)
async def post_decision(
    brief_id: str,
    payload: DecisionPayload,
    request: Request,
) -> DecisionResponse:
    """
    Submit a human review decision for a pending brief.
    Unblocks the human_gate_node in the LangGraph pipeline.
    """
    brief = get_brief(brief_id)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"Brief '{brief_id}' not found.")

    if payload.decision == "edited" and not payload.edits:
        raise HTTPException(
            status_code=422,
            detail="'edits' field is required when decision is 'edited'.",
        )

    reviewed_at = datetime.now(timezone.utc).isoformat()

    await set_decision(
        brief_id=brief_id,
        decision=payload.decision,
        edits=payload.edits,
        reviewer=payload.reviewer,
    )

    update_latest_draft_decision(
        brief_id=brief_id,
        human_decision=payload.decision,
        human_edits=payload.edits,
        reviewed_by=payload.reviewer,
        reviewed_at=reviewed_at,
    )

    if payload.decision == "rejected":
        update_brief_status(brief_id, "rejected")
        push_sync(brief_id, f"[HumanGate] Brief rejected by {payload.reviewer}.")
    else:
        update_brief_status(brief_id, "curating")
        push_sync(
            brief_id,
            f"[HumanGate] Decision '{payload.decision}' recorded - resuming pipeline.",
        )

    write_audit(
        brief_id=brief_id,
        event_type="human_decision",
        event_data={
            "decision": payload.decision,
            "reviewer": payload.reviewer,
            "has_edits": bool(payload.edits),
        },
        actor=payload.reviewer,
    )

    return DecisionResponse(
        brief_id=brief_id,
        decision=payload.decision,
        message=f"Decision '{payload.decision}' recorded and pipeline unblocked.",
    )
