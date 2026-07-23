from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import mlflow
from jinja2 import Environment, FileSystemLoader

from api.sse_queues import push_sync
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


# [rag-perf] Do NOT size this down to the channel's character limit. The draft
# prompt asks for strict JSON and tells the model to count characters before
# answering, which makes a reasoning model think at length before emitting any
# content. Measured on gpt-oss-120b: a 1624-token budget returned an EMPTY
# response for every channel, because reasoning consumed the whole allowance.
# 4096 is the smallest budget observed to reliably produce a draft.
_DRAFT_MAX_TOKENS = 4096


def _cached_context(
    state: AgentState,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    """[rag-perf] Return the context a previous revision already retrieved.
    None when nothing was retrieved yet, so the caller falls back to a real
    retrieval rather than handing the generator an empty context."""
    campaigns = state.get("retrieved_campaigns") or []
    social = state.get("retrieved_social") or []
    guidelines = state.get("retrieved_guidelines") or []
    if not campaigns and not social and not guidelines:
        return None
    return campaigns, social, guidelines


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

        # [ring-liveness] Emit progress the instant it happens. LangGraph flushes a
        # node's sse_events only when the node RETURNS, so the retrieval + draft
        # work (~10-15s) would otherwise freeze the progress ring on one step.
        # push_sync goes straight to the live SSE queue; delivered here instead of
        # the node-end flush, so there is no duplication.
        def live(msg: str) -> None:
            push_sync(brief_id, msg)
        persona_profile = constraints.get("persona_profile", {})

        brief_text = (
            f"Brand: {brand}. Channel: {channel}. "
            f"Persona: {persona_name}. Key message: {key_message}."
        )

        # [rag-perf] Built after char_limit is known so the token budget can be
        # sized to the channel — see _max_tokens_for_draft.
        llm = None

        loop = asyncio.get_event_loop()

        # [rag-perf] brief_text is derived only from brand/channel/persona/
        # key_message — none of which change between revisions — so a revision
        # would retrieve byte-identical chunks at the cost of a full HyDE+CRAG
        # round trip per collection. Reuse what revision 0 already fetched.
        cached = _cached_context(state) if revision_count > 0 else None

        if cached is not None:
            campaigns, social_content, guidelines = cached
            live(
                f"[Generator] Reusing retrieved context (revision {revision_count}) — "
                "the brief is unchanged, so retrieval is skipped."
            )
        else:
            live(f"[Generator] Starting retrieval (revision {revision_count})…")

            campaigns = await retrieve_with_hyde(
                brief_text=brief_text,
                collection_name=cc.APPROVED_CAMPAIGNS,
                top_k=5,
                crag_threshold=settings.CRAG_THRESHOLD,
            )

            social_content: list[dict[str, Any]] = []
            if channel == "social":
                social_content = await retrieve_with_hyde(
                    brief_text=brief_text,
                    collection_name=cc.SOCIAL_CONTENT,
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
                    top_k=3,
                    crag_threshold=0.3,
                )

            live(
                f"[Generator] Retrieved {len(campaigns)} campaigns, "
                f"{len(guidelines)} guidelines, {len(social_content)} social examples."
            )

        char_limit = _char_limit_for_channel(channel, persona_profile)
        min_chars = _min_chars_for_channel(channel, char_limit)

        llm = get_llm(
            temperature=0.8 if revision_count == 0 else 0.5,
            max_tokens=_DRAFT_MAX_TOKENS,
        )

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

        live("[Generator] Calling LLM for draft generation…")
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

        # [rag-perf] An empty completion is not a draft. A reasoning model that
        # spends its whole token budget thinking returns content="" with no API
        # error, which previously surfaced as a silent "0 chars" draft and sent
        # nothing downstream. Say so plainly instead.
        if not (response.content or "").strip():
            errors.append("Generator returned an empty completion (token budget exhausted?)")
            live(
                "[Generator] WARNING: model returned an empty response — "
                "no content to draft from."
            )

        live(f"[Generator] Draft generated ({len(draft_text)} chars).")

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
