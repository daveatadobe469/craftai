from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import mlflow
from jinja2 import Environment, FileSystemLoader

from config import get_llm, settings
from graph.state import AgentState
from rag import chroma_client as cc
from rag import embedder, evaluator
from rag.retriever import retrieve_with_hyde

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
_jinja_env = Environment(loader=FileSystemLoader(str(_PROMPTS_DIR)), autoescape=False)


def _parse_draft_json(raw: str) -> tuple[str, dict[str, Any]]:
    """
    Extract the JSON block from the LLM response.
    Returns (draft_text, metadata_dict).
    """
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        raw_json = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        raw_json = brace_match.group(0) if brace_match else raw

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw.strip(), {}

    body_keys = ["body", "copy", "content"]
    for key in body_keys:
        if key in data:
            draft_text = str(data[key])
            return draft_text, data

    draft_text = json.dumps(data, indent=2)
    return draft_text, data


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

        template = _jinja_env.get_template(f"{channel}.j2")
        prompt_text = template.render(
            brand=brand,
            persona=persona_name,
            persona_description=persona_profile.get("description", ""),
            key_message=key_message,
            preferred_tone=persona_profile.get("preferred_tone", "professional"),
            age_range=persona_profile.get("age_range", "25-45"),
            income_bracket=persona_profile.get("income_bracket", "middle"),
            interests=persona_profile.get("interests", []),
            pain_points=persona_profile.get("pain_points", []),
            char_limit_email=persona_profile.get("char_limit_email", 500),
            char_limit_social=persona_profile.get("char_limit_social", 280),
            char_limit_linkedin=persona_profile.get("char_limit_linkedin", 700),
            char_limit_ad=persona_profile.get("char_limit_ad", 150),
            char_limit_blog=persona_profile.get("char_limit_blog", 2000),
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

        draft_text, draft_metadata = _parse_draft_json(response.content)
        draft_metadata["channel"] = channel
        draft_metadata["brand"] = brand
        draft_metadata["persona"] = persona_name
        draft_metadata["revision_count"] = revision_count
        draft_metadata["generation_ms"] = generation_ms

        sse_events.append(f"[Generator] Draft generated ({len(draft_text)} chars). Running RAGAS…")

        context_texts = [c["document"] for c in campaigns + guidelines]
        ragas_scores = await evaluator.evaluate_ragas(
            question=brief_text,
            answer=draft_text,
            contexts=context_texts,
        )

        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("craftai_campaigns")
            with mlflow.start_run(run_name=brief_id, nested=False):
                mlflow.log_metrics({f"ragas_{k}": v for k, v in ragas_scores.items()})
                mlflow.log_metric("generation_ms", generation_ms)
                mlflow.log_metric("revision_count", revision_count)
        except Exception:
            pass

        sse_events.append(
            f"[Generator] RAGAS scores — "
            f"faithfulness: {ragas_scores.get('faithfulness', 0):.2f}, "
            f"relevancy: {ragas_scores.get('answer_relevancy', 0):.2f}"
        )

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
