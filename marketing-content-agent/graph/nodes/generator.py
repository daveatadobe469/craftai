from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import mlflow
from jinja2 import Environment, FileSystemLoader

from config import get_llm, settings
from graph.draft_parser import parse_and_format_draft
from graph.state import AgentState
from rag import chroma_client as cc
from rag import embedder, evaluator
from rag.retriever import retrieve_with_hyde

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
_jinja_env = Environment(loader=FileSystemLoader(str(_PROMPTS_DIR)), autoescape=False)


def _min_chars_for_channel(channel: str, char_limit: int) -> int:
    """Minimum target length so small models do not stop at one short sentence."""
    ch = channel.lower()
    if ch == "blog":
        return max(800, int(char_limit * 0.45))
    if ch == "social":
        return max(180, int(char_limit * 0.70))
    if ch == "ad":
        return max(90, int(char_limit * 0.75))
    if ch == "linkedin":
        return max(400, int(char_limit * 0.65))
    return max(280, int(char_limit * 0.65))


def _char_limit_for_channel(channel: str, persona_profile: dict[str, Any]) -> int:
    key = f"char_limit_{channel.lower()}"
    defaults = {
        "char_limit_email": 750,
        "char_limit_social": 280,
        "char_limit_linkedin": 1200,
        "char_limit_ad": 200,
        "char_limit_blog": 3500,
    }
    return int(persona_profile.get(key) or defaults.get(key, 750))


async def generator_node(state: AgentState) -> AgentState:
    """
    Stage 2 — Generator Node
    1. HyDE rewrite + CRAG retrieval from approved_campaigns & social_content.
    2. Build Jinja2 prompt with retrieved context.
    3. Call LLM, parse structured draft.
    4. RAGAS evaluation.
    5. Log scores to MLflow.
    """
    errors: list[str] = list(state.get("errors") or [])
    sse_events: list[str] = list(state.get("sse_events") or [])

    try:
        brief_id = state["brief_id"]
        brand = state["brand"]
        channel = state["channel"]
        persona_name = state["persona"]
        key_message = state["key_message"]
        constraints = state.get("constraints") or {}
        revision_count = state.get("revision_count") or 0
        rule_violations = state.get("rule_violations") or []
        persona_profile = constraints.get("persona_profile", {})

        brief_text = (
            f"Brand: {brand}. Channel: {channel}. "
            f"Persona: {persona_name}. Key message: {key_message}."
        )

        llm = get_llm(temperature=0.8 if revision_count == 0 else 0.5)

        sse_events.append(f"[Generator] Starting retrieval (revision {revision_count})…")

        loop = asyncio.get_event_loop()

        campaigns = await retrieve_with_hyde(
            brief_text=brief_text,
            collection_name=cc.APPROVED_CAMPAIGNS,
            llm=llm,
            top_k=5,
            crag_threshold=settings.CRAG_THRESHOLD,
        )

        social_content: list[dict[str, Any]] = []
        if channel == "social":
            social_content = await retrieve_with_hyde(
                brief_text=brief_text,
                collection_name=cc.SOCIAL_CONTENT,
                llm=llm,
                top_k=3,
                crag_threshold=settings.CRAG_THRESHOLD,
            )

        guidelines_count = await loop.run_in_executor(
            None, cc.collection_count, cc.BRAND_GUIDELINES
        )
        guidelines: list[dict[str, Any]] = []
        if guidelines_count > 0:
            guidelines = await retrieve_with_hyde(
                brief_text=brief_text,
                collection_name=cc.BRAND_GUIDELINES,
                llm=llm,
                top_k=3,
                crag_threshold=0.3,
            )

        sse_events.append(
            f"[Generator] Retrieved {len(campaigns)} campaigns, "
            f"{len(guidelines)} guidelines, {len(social_content)} social examples."
        )

        char_limit = _char_limit_for_channel(channel, persona_profile)
        min_chars = _min_chars_for_channel(channel, char_limit)

        template = _jinja_env.get_template(f"{channel}.j2")
        prompt_text = template.render(
            channel=channel,
            brand=brand,
            persona=persona_name,
            persona_description=persona_profile.get("description", ""),
            key_message=key_message,
            preferred_tone=persona_profile.get("preferred_tone", "professional"),
            age_range=persona_profile.get("age_range", "25-45"),
            income_bracket=persona_profile.get("income_bracket", "middle"),
            interests=persona_profile.get("interests", []),
            pain_points=persona_profile.get("pain_points", []),
            char_limit_email=persona_profile.get("char_limit_email", 750),
            char_limit_social=persona_profile.get("char_limit_social", 280),
            char_limit_linkedin=persona_profile.get("char_limit_linkedin", 1200),
            char_limit_ad=persona_profile.get("char_limit_ad", 200),
            char_limit_blog=persona_profile.get("char_limit_blog", 3500),
            min_chars=min_chars,
            max_chars=char_limit,
            guidelines=guidelines,
            campaigns=campaigns,
            social_content=social_content,
            revision_count=revision_count,
            rule_violations=rule_violations,
        )

        sse_events.append("[Generator] Calling LLM for draft generation…")
        t0 = time.monotonic()
        response = await loop.run_in_executor(None, llm.invoke, prompt_text)
        generation_ms = int((time.monotonic() - t0) * 1000)

        draft_text, draft_metadata = parse_and_format_draft(
            response.content, channel, char_limit
        )
        draft_metadata["channel"] = channel
        draft_metadata["brand"] = brand
        draft_metadata["persona"] = persona_name
        draft_metadata["revision_count"] = revision_count
        draft_metadata["generation_ms"] = generation_ms

        sse_events.append(f"[Generator] Draft generated ({len(draft_text)} chars).")

        # RAGAS moved out of this node: it cost ~4 LLM calls on EVERY revision,
        # which exhausted the API token quota and made the judge fail. It now runs
        # once, in the compliance node, on the draft that reaches the human gate.
        ragas_scores = state.get("ragas_scores") or {}

        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("craftai_campaigns")
            with mlflow.start_run(run_name=brief_id, nested=False):
                mlflow.log_metric("generation_ms", generation_ms)
                mlflow.log_metric("revision_count", revision_count)
        except Exception:
            pass

        return {
            **state,
            "draft": draft_text,
            "draft_metadata": draft_metadata,
            "retrieved_campaigns": campaigns,
            "retrieved_social": social_content,
            "retrieved_guidelines": guidelines,
            "ragas_scores": ragas_scores,
            "errors": errors,
            "sse_events": sse_events,
        }

    except Exception as exc:
        errors.append(f"Generator error: {exc}")
        sse_events.append(f"[Generator] ERROR: {exc}")
        return {**state, "errors": errors, "sse_events": sse_events}
