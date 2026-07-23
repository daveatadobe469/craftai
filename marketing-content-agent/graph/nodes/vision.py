# [image-based-campaign] Vision node (Stage 0, image → text).
# Runs before the orchestrator. No-op passthrough when the feature is off or no
# image was uploaded — so text-only briefs are unaffected and near-zero cost.
from __future__ import annotations

import asyncio

from config import settings
from graph.state import AgentState
from services.vision import describe_image


def _augment_key_message(key_message: str, description: str) -> str:
    """Fold the image description into the brief so downstream nodes need no changes."""
    return f"{key_message}\n\n[Reference image] {description}"


async def vision_node(state: AgentState) -> AgentState:
    errors: list[str] = list(state.get("errors") or [])
    sse_events: list[str] = list(state.get("sse_events") or [])

    image_ref = state.get("input_image_ref")
    # [image-based-campaign] Skip entirely unless enabled AND an image was provided.
    # [image-based-campaign] Only runs when a reference image was actually uploaded.
    # Independent of generate_image, which controls image OUTPUT, not input.
    if not image_ref:
        return {**state, "errors": errors, "sse_events": sse_events}

    try:
        sse_events.append("[Vision] Analysing uploaded image…")
        loop = asyncio.get_event_loop()
        with open(image_ref, "rb") as f:
            image_bytes = f.read()
        description = await loop.run_in_executor(None, describe_image, image_bytes)

        key_message = (state.get("key_message") or "").strip()
        augmented = _augment_key_message(key_message, description) if key_message else description

        sse_events.append(f"[Vision] Image described ({len(description)} chars).")
        return {
            **state,
            "key_message": augmented,
            "image_description": description,
            "errors": errors,
            "sse_events": sse_events,
        }
    except Exception as exc:
        # [image-based-campaign] Fail soft — keep the text pipeline running.
        errors.append(f"Vision error: {exc}")
        sse_events.append(f"[Vision] ERROR (continuing text-only): {exc}")
        return {**state, "errors": errors, "sse_events": sse_events}
