from __future__ import annotations

import asyncio
from typing import Any

import mlflow

from compliance.judge import score_draft
from compliance.tools import run_all_checks
from config import get_judge_llm, settings
from db.sqlite import update_brief_status, write_draft
from graph.state import AgentState


def route(state: AgentState) -> str:
    """
    Conditional edge router after compliance_node.
    Returns "revise" if non-compliant and under MAX_REVISIONS, else "gate".
    """
    revision_count = state.get("revision_count") or 0
    compliance_pass = state.get("compliance_pass", False)

    if not compliance_pass and revision_count < settings.MAX_REVISIONS:
        return "revise"
    return "gate"


async def compliance_node(state: AgentState) -> AgentState:
    """
    Stage 3 — Compliance Node
    Pass 1: Deterministic rule tools (no LLM).
    Pass 2: LLM-as-judge for holistic brand alignment.
    Sets compliance_pass based on violations + judge_score.
    """
    errors: list[str] = list(state.get("errors") or [])
    sse_events: list[str] = list(state.get("sse_events") or [])

    try:
        brief_id = state["brief_id"]
        draft = state.get("draft", "")
        channel = state["channel"]
        brand = state["brand"]
        constraints = state.get("constraints") or {}
        revision_count = state.get("revision_count") or 0

        if not draft:
            errors.append("Compliance: no draft to check.")
            return {
                **state,
                "compliance_pass": False,
                "rule_violations": ["No draft generated."],
                "judge_score": 0.0,
                "judge_evidence": "No draft.",
                "errors": errors,
                "sse_events": sse_events,
            }

        sse_events.append("[Compliance] Running deterministic rule checks…")

        persona_profile = constraints.get("persona_profile", {})
        persona_limits: dict[str, int] = {
            k: v for k, v in persona_profile.items() if k.startswith("char_limit_")
        }

        violations = run_all_checks(
            draft=draft,
            channel=channel,
            brand=brand,
            persona_limits=persona_limits if persona_limits else None,
            extra_required=None,
            allowed_domains=None,
            draft_metadata=state.get("draft_metadata") or {},
        )

        sse_events.append(
            f"[Compliance] Rule check complete — {len(violations)} violation(s) found."
        )

        retrieved_guidelines = state.get("retrieved_guidelines") or []
        guidelines_context = "\n".join(
            g.get("document", "") for g in retrieved_guidelines[:3]
        )

        llm = get_judge_llm(temperature=0.1)
        sse_events.append("[Compliance] Running LLM-as-judge scoring…")

        loop = asyncio.get_event_loop()
        judge_score, judge_evidence = await score_draft(
            draft=draft,
            channel=channel,
            brand=brand,
            guidelines_context=guidelines_context,
            llm=llm,
        )

        compliance_pass = (
            len(violations) == 0 and judge_score >= settings.JUDGE_THRESHOLD
        )

        sse_events.append(
            f"[Compliance] Judge score: {judge_score:.2f} "
            f"(threshold: {settings.JUDGE_THRESHOLD}). "
            f"Pass: {compliance_pass}."
        )

        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("craftai_campaigns")
            with mlflow.start_run(run_name=brief_id, nested=False):
                mlflow.log_metrics({
                    "judge_score": judge_score,
                    "rule_violations_count": len(violations),
                    "compliance_pass": int(compliance_pass),
                })
        except Exception:
            pass

        new_revision_count = revision_count + (0 if compliance_pass else 1)

        # Persist draft to SQLite NOW so the status endpoint can read it
        # at the human-gate stage (before the curator runs after approval).
        will_gate = compliance_pass or (new_revision_count >= settings.MAX_REVISIONS)
        # Merge violations + judge evidence into metadata so status endpoint
        # can surface them without a DB schema change.
        draft_metadata = dict(state.get("draft_metadata") or {})
        draft_metadata["rule_violations"]  = violations
        draft_metadata["judge_evidence"]   = judge_evidence

        try:
            await loop.run_in_executor(
                None,
                lambda: write_draft(
                    brief_id=brief_id,
                    content=draft,
                    revision_count=revision_count,
                    metadata=draft_metadata,
                    judge_score=judge_score,
                    compliance_pass=compliance_pass,
                    ragas_scores=state.get("ragas_scores") or {},
                    mlflow_run_id=state.get("mlflow_run_id") or "",
                ),
            )
            await loop.run_in_executor(
                None,
                update_brief_status,
                brief_id,
                "awaiting_review" if will_gate else "processing",
            )
        except Exception as db_exc:
            errors.append(f"Compliance DB write error: {db_exc}")

        return {
            **state,
            "rule_violations": violations,
            "judge_score": judge_score,
            "judge_evidence": judge_evidence,
            "compliance_pass": compliance_pass,
            "revision_count": new_revision_count,
            "errors": errors,
            "sse_events": sse_events,
        }

    except Exception as exc:
        errors.append(f"Compliance error: {exc}")
        sse_events.append(f"[Compliance] ERROR: {exc}")
        return {
            **state,
            "compliance_pass": False,
            "rule_violations": [f"Internal error: {exc}"],
            "judge_score": 0.0,
            "judge_evidence": f"error: {exc}",
            "errors": errors,
            "sse_events": sse_events,
        }
