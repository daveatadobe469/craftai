# [image-based-campaign] Art-director node (text → image).
# Runs on the compliance "gate" branch, just before the human gate, so it fires
# once per brief (not on every revision). No-op passthrough when the feature is
# off — the graph shape is identical to the text-only pipeline.
from __future__ import annotations

import asyncio
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from config import settings
from db.sqlite import update_latest_draft_metadata
from graph.state import AgentState
from services.image_generator import generate_image
from services.image_store import get_image_store

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
_jinja_env = Environment(loader=FileSystemLoader(str(_PROMPTS_DIR)), autoescape=False)


def _build_image_prompt(state: AgentState) -> str:
    """Render the art-direction prompt from brief + draft + guidelines."""
    guidelines = state.get("retrieved_guidelines") or []
    template = _jinja_env.get_template("image_prompt.j2")
    return template.render(
        brand=state.get("brand", ""),
        channel=state.get("channel", ""),
        key_message=state.get("key_message", ""),
        draft=state.get("draft", ""),
        guidelines=[g.get("document", "") for g in guidelines[:3]],
    ).strip()


async def art_director_node(state: AgentState) -> AgentState:
    errors: list[str] = list(state.get("errors") or [])
    sse_events: list[str] = list(state.get("sse_events") or [])

    # [image-based-campaign] Skip unless enabled and there is a draft to illustrate.
    if not settings.IMAGE_FEATURE_ENABLED or not state.get("draft"):
        return {**state, "errors": errors, "sse_events": sse_events}

    try:
        brief_id = state["brief_id"]
        loop = asyncio.get_event_loop()

        prompt = _build_image_prompt(state)
        sse_events.append("[ArtDirector] Generating campaign image…")

        image_bytes = await loop.run_in_executor(None, generate_image, prompt)
        ref = await loop.run_in_executor(
            None, lambda: get_image_store().save(image_bytes, "images", brief_id)
        )

        # [image-based-campaign] Persist the pointer into the existing draft metadata
        # (no schema change) so /status and the UI can surface the image.
        await loop.run_in_executor(
            None, update_latest_draft_metadata, brief_id,
            {"image_url": ref.url, "image_path": ref.path},
        )

        # [image-based-campaign] Also merge into state's draft_metadata: the curator
        # writes a NEW draft row from this dict after approval, and get_latest_draft
        # returns that newest row — without this the image ref is lost post-approval.
        draft_metadata = dict(state.get("draft_metadata") or {})
        draft_metadata["image_url"] = ref.url
        draft_metadata["image_path"] = ref.path

        sse_events.append(f"[ArtDirector] Image ready: {ref.url}")
        return {
            **state,
            "draft_metadata": draft_metadata,
            "image_prompt": prompt,
            "image_path": ref.path,
            "image_url": ref.url,
            "errors": errors,
            "sse_events": sse_events,
        }
    except Exception as exc:
        # [image-based-campaign] Fail soft — the text draft still goes to review.
        errors.append(f"ArtDirector error: {exc}")
        sse_events.append(f"[ArtDirector] ERROR (draft still proceeds): {exc}")
        return {**state, "errors": errors, "sse_events": sse_events}
