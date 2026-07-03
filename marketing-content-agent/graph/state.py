from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # ── Identity ──────────────────────────────────────────────────────────────
    brief_id: str
    brand: str
    channel: Literal["email", "linkedin", "social", "ad", "blog"]
    persona: str
    key_message: str
    constraints: dict[str, Any]

    # ── RAG context ───────────────────────────────────────────────────────────
    retrieved_campaigns: list[dict[str, Any]]
    retrieved_social: list[dict[str, Any]]
    retrieved_guidelines: list[dict[str, Any]]

    # ── Generation ────────────────────────────────────────────────────────────
    draft: str
    draft_metadata: dict[str, Any]
    revision_count: int

    # ── Compliance ────────────────────────────────────────────────────────────
    rule_violations: list[str]
    judge_score: float
    judge_evidence: str
    compliance_pass: bool

    # ── Human gate ────────────────────────────────────────────────────────────
    human_decision: Optional[Literal["approved", "edited", "rejected"]]
    human_edits: Optional[str]
    reviewed_by: Optional[str]
    reviewed_at: Optional[datetime]

    # ── Curator ───────────────────────────────────────────────────────────────
    indexed_doc_id: Optional[str]

    # ── Evaluation & observability ────────────────────────────────────────────
    ragas_scores: dict[str, float]
    mlflow_run_id: str

    # ── Orchestration ─────────────────────────────────────────────────────────
    plan: list[str]
    errors: list[str]
    sse_events: list[str]
