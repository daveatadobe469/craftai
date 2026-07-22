from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db.sqlite import get_audit_event_data, get_brief, get_latest_draft

router = APIRouter()


class StatusResponse(BaseModel):
    brief_id: str
    status: str
    brand: str = ""
    channel: str = ""
    persona: str = ""
    draft: str = ""
    draft_metadata: dict = {}
    judge_score: float = 0.0
    judge_evidence: str = ""
    compliance_pass: bool = False
    rule_violations: list[str] = []
    human_decision: str | None = None
    ragas_scores: dict = {}
    indexed_doc_id: str | None = None
    errors: list[str] = []


@router.get("/status/{brief_id}", response_model=StatusResponse)
async def get_status(brief_id: str) -> StatusResponse:
    """Return current pipeline status and latest draft for a brief_id."""
    brief = get_brief(brief_id)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"Brief '{brief_id}' not found.")

    draft_row = get_latest_draft(brief_id)
    meta: dict = draft_row.get("metadata", {}) if draft_row else {}

    draft_content = draft_row.get("content", "") if draft_row else ""
    human_edits = draft_row.get("human_edits") if draft_row else None
    if human_edits and str(human_edits).strip():
        draft_content = str(human_edits).strip()

    indexed = get_audit_event_data(brief_id, "content_indexed") or {}
    indexed_doc_id = indexed.get("doc_base_id")

    return StatusResponse(
        brief_id=brief_id,
        status=brief.get("status", "pending"),
        brand=brief.get("brand", ""),
        channel=brief.get("channel", ""),
        persona=brief.get("persona", ""),
        draft=draft_content,
        draft_metadata=meta,
        judge_score=draft_row.get("judge_score") or 0.0 if draft_row else 0.0,
        judge_evidence=meta.get("judge_evidence", ""),
        compliance_pass=draft_row.get("compliance_pass", False) if draft_row else False,
        rule_violations=meta.get("rule_violations", []),
        human_decision=draft_row.get("human_decision") if draft_row else None,
        ragas_scores=draft_row.get("ragas_scores", {}) if draft_row else {},
        indexed_doc_id=indexed_doc_id,
        errors=[],
    )
