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
from services.image_judge import ImageVerdict, judge_image
from services.image_store import _sniff_ext, get_image_store  # noqa: PLC2701

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


_MIME_BY_EXT = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


async def _judge(state: AgentState, image_bytes: bytes) -> ImageVerdict | None:
    """[image-based-campaign] Grade the generated image with the cross-vendor judge.
    Returns None when the judge is switched off."""
    if not settings.IMAGE_JUDGE_ENABLED:
        return None

    guidelines = [g.get("document", "") for g in (state.get("retrieved_guidelines") or [])[:3]]
    mime = _MIME_BY_EXT.get(_sniff_ext(image_bytes), "image/png")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: judge_image(
            image_bytes=image_bytes,
            mime=mime,
            brand=state.get("brand", ""),
            channel=state.get("channel", ""),
            draft=state.get("draft", ""),
            guidelines=guidelines,
        ),
    )


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

        sse_events.append(f"[ArtDirector] Image ready: {ref.url}")

        # [image-based-campaign] Cross-vendor compliance check on the visual.
        extra: dict[str, Any] = {"image_url": ref.url, "image_path": ref.path}
        verdict = await _judge(state, image_bytes)
        if verdict is not None:
            extra["image_judge_score"] = verdict.score
            extra["image_judge_evidence"] = verdict.evidence
            extra["image_judge_issues"] = verdict.issues
            if verdict.unavailable:
                sse_events.append(f"[ImageJudge] UNAVAILABLE — {verdict.evidence}")
            else:
                passed = verdict.score >= settings.IMAGE_JUDGE_THRESHOLD
                sse_events.append(
                    f"[ImageJudge] Score {verdict.score:.2f} "
                    f"(threshold {settings.IMAGE_JUDGE_THRESHOLD}) — "
                    f"{'PASS' if passed else 'FAIL'}, {len(verdict.issues)} issue(s)."
                )

        # [image-based-campaign] Persist into the existing draft metadata (no schema
        # change) so /status and the UI can surface the image and its verdict.
        await loop.run_in_executor(
            None, update_latest_draft_metadata, brief_id, extra,
        )

        # [image-based-campaign] Also merge into state's draft_metadata: the curator
        # writes a NEW draft row from this dict after approval, and get_latest_draft
        # returns that newest row — without this the refs are lost post-approval.
        draft_metadata = dict(state.get("draft_metadata") or {})
        draft_metadata.update(extra)

        return {
            **state,
            "draft_metadata": draft_metadata,
            "image_prompt": prompt,
            "image_path": ref.path,
            "image_url": ref.url,
            "image_judge_score": verdict.score if verdict else None,
            "image_judge_evidence": verdict.evidence if verdict else "",
            "image_judge_issues": verdict.issues if verdict else [],
            "errors": errors,
            "sse_events": sse_events,
        }
    except Exception as exc:
        # [image-based-campaign] Fail soft — the text draft still goes to review.
        errors.append(f"ArtDirector error: {exc}")
        sse_events.append(f"[ArtDirector] ERROR (draft still proceeds): {exc}")
        return {**state, "errors": errors, "sse_events": sse_events}
