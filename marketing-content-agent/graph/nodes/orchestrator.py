from __future__ import annotations

import asyncio
import logging
from typing import Any

import mlflow

from config import settings
from db.sqlite import get_persona, write_audit
from graph.state import AgentState

logger = logging.getLogger(__name__)

_VALID_CHANNELS = {"email", "linkedin", "social", "ad", "blog"}


async def orchestrator_node(state: AgentState) -> AgentState:
    """
    Stage 1 — Orchestrator Node
    - Validates the brief fields.
    - Loads persona profile from SQLite.
    - Initialises MLflow run.
    - Builds a 3-step ReAct plan.
    - Appends SSE event.
    """
    errors: list[str] = list(state.get("errors") or [])
    sse_events: list[str] = list(state.get("sse_events") or [])

    try:
        brief_id = state["brief_id"]
        brand = state.get("brand", "").strip()
        channel = state.get("channel", "").strip().lower()
        persona_name = state.get("persona", "").strip()
        key_message = state.get("key_message", "").strip()
        # campaign_type may arrive top-level or nested in constraints; default safely.
        campaign_type = (
            state.get("campaign_type")
            or (state.get("constraints") or {}).get("campaign_type")
            or "general"
        )

        if not brand:
            raise ValueError("'brand' field is required and cannot be empty.")
        if channel not in _VALID_CHANNELS:
            raise ValueError(
                f"Invalid channel '{channel}'. Must be one of: {', '.join(sorted(_VALID_CHANNELS))}."
            )
        if not persona_name:
            raise ValueError("'persona' field is required and cannot be empty.")
        if not key_message:
            raise ValueError("'key_message' field is required and cannot be empty.")

        loop = asyncio.get_event_loop()
        persona_profile = await loop.run_in_executor(None, get_persona, persona_name)
        if persona_profile is None:
            logger.warning(
                "Persona %r not found in SQLite — using generic default profile. "
                "Content quality may degrade; ensure personas are seeded "
                "(check db/persona_seed.py / migrations).",
                persona_name,
            )
            sse_events.append(
                f"[Orchestrator] WARNING: persona '{persona_name}' not found — "
                "using generic default profile."
            )
            persona_profile = {
                "name": persona_name,
                "description": f"A {persona_name} customer segment.",
                "age_range": "25-45",
                "income_bracket": "middle",
                "interests": [],
                "pain_points": [],
                "preferred_tone": "professional",
                "char_limit_email": 750,
                "char_limit_social": 280,
                "char_limit_linkedin": 1200,
                "char_limit_ad": 200,
                "char_limit_blog": 3500,
            }

        constraints = dict(state.get("constraints") or {})
        constraints["persona_profile"] = persona_profile

        mlflow_run_id = ""
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("craftai_campaigns")
            active_run = mlflow.start_run(run_name=brief_id, nested=False)
            mlflow_run_id = active_run.info.run_id
            mlflow.log_params({
                "brief_id": brief_id,
                "brand": brand,
                "channel": channel,
                "persona": persona_name,
                "key_message": key_message[:250],
            })
        except Exception:
            pass

        # Static plan — no LLM call needed here; downstream nodes don't use this value
        # and the ReAct prompt was adding an unnecessary full LLM round-trip (~15-30s).
        plan_steps = [
            "Thought: Analyse brief and retrieve relevant brand context.",
            "Action: Generate channel-specific draft using RAG and persona constraints.",
            "Observation: Compliance-checked and human-approved draft ready for indexing.",
        ]

        await loop.run_in_executor(
            None,
            write_audit,
            brief_id,
            "orchestrator_complete",
            {"plan_steps": len(plan_steps), "mlflow_run_id": mlflow_run_id},
            "orchestrator",
        )

        sse_events.append(
            f"[Orchestrator] Brief validated. Persona: {persona_name}. "
            f"Channel: {channel}. Plan ready ({len(plan_steps)} steps)."
        )

        try:
            mlflow.end_run()
        except Exception:
            pass

        return {
            **state,
            "brand": brand,
            "channel": channel,
            "persona": persona_name,
            "key_message": key_message,
            "campaign_type": campaign_type,
            "constraints": constraints,
            "plan": plan_steps,
            "revision_count": state.get("revision_count") or 0,
            "rule_violations": [],
            "errors": errors,
            "sse_events": sse_events,
            "mlflow_run_id": mlflow_run_id,
            "ragas_scores": {},
            "retrieved_campaigns": [],
            "retrieved_social": [],
            "retrieved_guidelines": [],
        }

    except Exception as exc:
        errors.append(f"Orchestrator error: {exc}")
        sse_events.append(f"[Orchestrator] ERROR: {exc}")
        return {**state, "errors": errors, "sse_events": sse_events}
