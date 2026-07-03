from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.schemas.decision import DecisionPayload
from db.sqlite import get_brief, write_audit
from graph.nodes.human_gate import set_decision

router = APIRouter()


class DecisionResponse(BaseModel):
    brief_id: str
    decision: str
    message: str


@router.post("/decision/{brief_id}", response_model=DecisionResponse)
async def post_decision(brief_id: str, payload: DecisionPayload) -> DecisionResponse:
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

    set_decision(
        brief_id=brief_id,
        decision=payload.decision,
        edits=payload.edits,
        reviewer=payload.reviewer,
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
