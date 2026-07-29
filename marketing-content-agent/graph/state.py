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
    campaign_type: str
    constraints: dict[str, Any]

    # ── RAG context ───────────────────────────────────────────────────────────
    retrieved_campaigns: list[dict[str, Any]]
    retrieved_social: list[dict[str, Any]]
    retrieved_guidelines: list[dict[str, Any]]

    # ── [image-based-campaign] Image feature ──────────────────────────────────
    # Per-brief choice: does this campaign want a generated visual? Set from the
    # brief form / API payload. False = the draft comes back as text only.
    generate_image: bool
    # Input (image → text): path to an uploaded image + its vision description.
    input_image_ref: Optional[str]
    image_description: Optional[str]
    # Output (text → image): the art-direction prompt + stored image reference.
    image_prompt: Optional[str]
    image_path: Optional[str]
    image_url: Optional[str]
    # Cross-vendor image compliance judge. Score is None when it could not run.
    image_judge_score: Optional[float]
    image_judge_evidence: Optional[str]
    image_judge_issues: list[str]

    # ── Generation ────────────────────────────────────────────────────────────
    draft: str
    draft_metadata: dict[str, Any]
    revision_count: int

    # ── Compliance ────────────────────────────────────────────────────────────
    rule_violations: list[str]
    judge_score: float
    judge_evidence: str
    compliance_pass: bool
    # True when the LLM judge could not be evaluated (API/rate-limit/parse error).
    # Fail-open: route straight to the human gate without counting a revision.
    judge_unavailable: bool

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
