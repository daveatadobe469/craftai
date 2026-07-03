from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from config import settings

router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────────────────

class AuditEvent(BaseModel):
    id: int
    brief_id: str
    event_type: str
    event_data: dict[str, Any]
    actor: str
    ts: str


class AuditResponse(BaseModel):
    total: int
    page: int
    page_size: int
    events: list[AuditEvent]


class BriefSummary(BaseModel):
    brief_id: str
    brand: str
    channel: str
    persona: str
    key_message: str
    status: str
    created_at: str
    updated_at: str


class DraftSummary(BaseModel):
    draft_id: str
    brief_id: str
    revision_count: int
    judge_score: Optional[float]
    compliance_pass: bool
    human_decision: Optional[str]
    reviewed_by: Optional[str]
    reviewed_at: Optional[str]
    created_at: str


class PipelineStats(BaseModel):
    total_briefs: int
    total_drafts: int
    total_audit_events: int
    approved_count: int
    rejected_count: int
    pending_count: int
    avg_judge_score: float
    avg_revisions: float
    events_by_type: dict[str, int]


# ── Helper ────────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/audit/events", response_model=AuditResponse)
async def get_audit_events(
    brief_id: Optional[str] = Query(None, description="Filter by brief_id"),
    event_type: Optional[str] = Query(None, description="Filter by event_type"),
    actor: Optional[str] = Query(None, description="Filter by actor"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> AuditResponse:
    """Return paginated audit log events with optional filters."""
    conditions: list[str] = []
    params: list[Any] = []

    if brief_id:
        conditions.append("brief_id = ?")
        params.append(brief_id)
    if event_type:
        conditions.append("event_type = ?")
        params.append(event_type)
    if actor:
        conditions.append("actor = ?")
        params.append(actor)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    offset = (page - 1) * page_size

    try:
        with _connect() as conn:
            total_row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM audit_log {where}", params
            ).fetchone()
            total = total_row["cnt"] if total_row else 0

            rows = conn.execute(
                f"SELECT * FROM audit_log {where} ORDER BY ts DESC LIMIT ? OFFSET ?",
                params + [page_size, offset],
            ).fetchall()
    except sqlite3.OperationalError:
        return AuditResponse(total=0, page=page, page_size=page_size, events=[])

    events = [
        AuditEvent(
            id=row["id"],
            brief_id=row["brief_id"],
            event_type=row["event_type"],
            event_data=json.loads(row["event_data"] or "{}"),
            actor=row["actor"],
            ts=row["ts"],
        )
        for row in rows
    ]

    return AuditResponse(total=total, page=page, page_size=page_size, events=events)


@router.get("/audit/briefs", response_model=list[BriefSummary])
async def get_briefs(
    status: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    channel: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> list[BriefSummary]:
    """Return a list of submitted briefs with optional filters."""
    conditions: list[str] = []
    params: list[Any] = []

    if status:
        conditions.append("status = ?")
        params.append(status)
    if brand:
        conditions.append("brand = ?")
        params.append(brand)
    if channel:
        conditions.append("channel = ?")
        params.append(channel)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    try:
        with _connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM briefs {where} ORDER BY created_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        BriefSummary(
            brief_id=row["brief_id"],
            brand=row["brand"],
            channel=row["channel"],
            persona=row["persona"],
            key_message=row["key_message"][:120] + ("…" if len(row["key_message"]) > 120 else ""),
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


@router.get("/audit/drafts", response_model=list[DraftSummary])
async def get_drafts(
    brief_id: Optional[str] = Query(None),
    human_decision: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> list[DraftSummary]:
    """Return draft records with optional filters."""
    conditions: list[str] = []
    params: list[Any] = []

    if brief_id:
        conditions.append("brief_id = ?")
        params.append(brief_id)
    if human_decision:
        conditions.append("human_decision = ?")
        params.append(human_decision)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    try:
        with _connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM drafts {where} ORDER BY created_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        DraftSummary(
            draft_id=row["draft_id"],
            brief_id=row["brief_id"],
            revision_count=row["revision_count"],
            judge_score=row["judge_score"],
            compliance_pass=bool(row["compliance_pass"]),
            human_decision=row["human_decision"],
            reviewed_by=row["reviewed_by"],
            reviewed_at=row["reviewed_at"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


@router.get("/audit/stats", response_model=PipelineStats)
async def get_pipeline_stats() -> PipelineStats:
    """Return aggregate statistics across the entire pipeline."""
    try:
        with _connect() as conn:
            total_briefs = conn.execute("SELECT COUNT(*) FROM briefs").fetchone()[0]
            total_drafts = conn.execute("SELECT COUNT(*) FROM drafts").fetchone()[0]
            total_events = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]

            approved = conn.execute(
                "SELECT COUNT(*) FROM drafts WHERE human_decision='approved'"
            ).fetchone()[0]
            rejected = conn.execute(
                "SELECT COUNT(*) FROM drafts WHERE human_decision='rejected'"
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM briefs WHERE status='pending'"
            ).fetchone()[0]

            avg_score_row = conn.execute(
                "SELECT AVG(judge_score) FROM drafts WHERE judge_score IS NOT NULL"
            ).fetchone()
            avg_score = float(avg_score_row[0] or 0.0)

            avg_rev_row = conn.execute(
                "SELECT AVG(revision_count) FROM drafts"
            ).fetchone()
            avg_revisions = float(avg_rev_row[0] or 0.0)

            event_rows = conn.execute(
                "SELECT event_type, COUNT(*) as cnt FROM audit_log GROUP BY event_type"
            ).fetchall()
            events_by_type = {row[0]: row[1] for row in event_rows}

    except sqlite3.OperationalError:
        return PipelineStats(
            total_briefs=0, total_drafts=0, total_audit_events=0,
            approved_count=0, rejected_count=0, pending_count=0,
            avg_judge_score=0.0, avg_revisions=0.0, events_by_type={},
        )

    return PipelineStats(
        total_briefs=total_briefs,
        total_drafts=total_drafts,
        total_audit_events=total_events,
        approved_count=approved,
        rejected_count=rejected,
        pending_count=pending,
        avg_judge_score=round(avg_score, 3),
        avg_revisions=round(avg_revisions, 2),
        events_by_type=events_by_type,
    )


@router.get("/audit/event-types")
async def get_event_types() -> dict[str, list[str]]:
    """Return distinct event_type and actor values for filter dropdowns."""
    try:
        with _connect() as conn:
            types = [r[0] for r in conn.execute(
                "SELECT DISTINCT event_type FROM audit_log ORDER BY event_type"
            ).fetchall()]
            actors = [r[0] for r in conn.execute(
                "SELECT DISTINCT actor FROM audit_log ORDER BY actor"
            ).fetchall()]
    except sqlite3.OperationalError:
        return {"event_types": [], "actors": []}

    return {"event_types": types, "actors": actors}
