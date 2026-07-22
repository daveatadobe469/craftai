# [image-based-campaign] Image compliance judge — grades a GENERATED campaign
# image against brand guidelines + the approved copy.
#
# Provider is chosen by IMAGE_JUDGE_PROVIDER and must be a DIFFERENT vendor from
# IMAGE_PROVIDER, so the model that generated the image never grades its own work.
#   "groq" — qwen3.6-27b vision (free tier, reuses GROQ_API_KEY); cross-vendor
#            relative to Cloudflare/Flux and Google/Gemini image generation.
#
# KNOWN LIMITATION: Groq's free tier caps this model at 8,000 tokens/minute, and
# it is a reasoning model — image (~2.6k tokens) + reasoning + JSON answer does not
# reliably fit, so it succeeded in only 3 of 7 trial runs. IMAGE_JUDGE_ENABLED
# therefore defaults to false and generated images are reviewed by a human only.
# Add a provider branch below if a higher-throughput vision model becomes available.
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field

from config import settings
from services.vision import strip_think  # shared: same reasoning model

logger = logging.getLogger(__name__)

# Groq's free tier caps this vision model at 8,000 tokens/MINUTE, and `max_tokens`
# counts toward the request size. A 1024px image is ~2.6k tokens, so the reply
# budget must stay well under the cap or the request is rejected 413 before it runs.
_GROQ_MAX_TOKENS = 4096

_SYSTEM = (
    "You are a brand-compliance reviewer for marketing visuals. You are grading an "
    "AI-generated campaign image. Judge only what is actually visible in the image — "
    "do not assume intent. Score 0.0 (unusable) to 1.0 (fully compliant) on: brand-guideline "
    "adherence, absence of prohibited or off-brand imagery, whether any text rendered IN the "
    "image is legible and spelled correctly, and whether that text matches the approved copy. "
    "Garbled, duplicated, or misspelled in-image text is a serious defect — call it out "
    "explicitly with the exact characters you see."
)

@dataclass
class ImageVerdict:
    """Judge outcome. `score is None` means the judge could not run (fail open)."""
    score: float | None
    evidence: str
    issues: list[str] = field(default_factory=list)

    @property
    def unavailable(self) -> bool:
        return self.score is None


def _user_prompt(brand: str, channel: str, draft: str, guidelines: list[str]) -> str:
    rules = "\n".join(f"- {g}" for g in guidelines) or "- No specific guidelines retrieved."
    return (
        f"Brand: {brand}\nChannel: {channel}\n\n"
        f"Brand guidelines:\n{rules}\n\n"
        f"Approved copy this visual must support:\n{draft}\n\n"
        "Grade the attached image. Respond with JSON only: "
        '{"score": <0.0-1.0>, "verdict": "pass"|"fail", "issues": [<string>], "evidence": "<one paragraph>"}'
    )


def _verdict_from_data(data: dict) -> ImageVerdict:
    score = min(1.0, max(0.0, float(data.get("score", 0.0))))
    return ImageVerdict(
        score=score,
        evidence=str(data.get("evidence", "")),
        issues=[str(i) for i in (data.get("issues") or [])],
    )


def _parse_json_verdict(raw: str) -> ImageVerdict:
    """Extract the JSON object from a free-form model reply (Groq has no
    structured-output guarantee, so tolerate fences, prose, and <think> blocks)."""
    text = strip_think(raw or "")
    if not text:
        raise ValueError("empty reply after stripping reasoning (likely truncated — raise max_tokens)")
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    return _verdict_from_data(json.loads(text))


# ── Groq vision (default) ─────────────────────────────────────────────────────
def _judge_groq(image_bytes: bytes, mime: str, prompt: str) -> ImageVerdict:
    from langchain_core.messages import HumanMessage, SystemMessage

    from config import get_vision_llm

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]),
    ]
    # The Groq vision model reasons before answering. json_mode gives a clean object,
    # but if it spends the whole budget inside <think> Groq rejects the empty result
    # with json_validate_failed — so fall back to free-form + our own parser.
    try:
        llm = get_vision_llm(temperature=0.1, max_tokens=_GROQ_MAX_TOKENS, json_mode=True)
        return _parse_json_verdict(llm.invoke(messages).content)
    except Exception as exc:
        if "json_validate_failed" not in str(exc):
            raise
        logger.info("json_mode produced no object; retrying free-form.")
        llm = get_vision_llm(temperature=0.1, max_tokens=_GROQ_MAX_TOKENS, json_mode=False)
        return _parse_json_verdict(llm.invoke(messages).content)


def judge_image(
    image_bytes: bytes,
    mime: str,
    brand: str,
    channel: str,
    draft: str,
    guidelines: list[str],
) -> ImageVerdict:
    """Grade a generated image. Blocking — call via executor. Never raises."""
    prompt = _user_prompt(brand, channel, draft, guidelines)
    provider = settings.IMAGE_JUDGE_PROVIDER
    try:
        if provider == "groq":
            return _judge_groq(image_bytes, mime, prompt)
        return ImageVerdict(None, f"unknown IMAGE_JUDGE_PROVIDER: {provider!r}")
    except Exception as exc:  # noqa: BLE001 — judge must never break the pipeline
        # Fail open: a rate limit or parse error is NOT evidence of non-compliance.
        logger.warning("Image judge (%s) failed: %s", provider, exc, exc_info=True)
        return ImageVerdict(None, f"image judge unavailable: {exc}")
