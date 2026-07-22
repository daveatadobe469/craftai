# [image-based-campaign] Vision service (image → text): describe an uploaded
# image so the downstream text pipeline can ground a campaign in it.
from __future__ import annotations

import base64
import re

from langchain_core.messages import HumanMessage

from config import get_vision_llm


def strip_think(text: str) -> str:
    """[image-based-campaign] The Groq vision model reasons out loud first. Drop the
    <think>…</think> block so callers get the answer, not the model's scratchpad.
    An unterminated block means the reply was truncated mid-reasoning."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL).strip()

_DESCRIBE_PROMPT = (
    "You are a marketing creative analyst. Describe this image for a campaign brief. "
    "Cover: the product or subject, visible brand cues, mood/emotion, colour palette, "
    "composition/style, and any text shown. Be concise (4-6 sentences), factual, no preamble."
)


def _data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def describe_image(image_bytes: bytes, mime: str = "image/png") -> str:
    """Return a text description of the image. Blocking — call via executor."""
    llm = get_vision_llm()
    message = HumanMessage(content=[
        {"type": "text", "text": _DESCRIBE_PROMPT},
        {"type": "image_url", "image_url": {"url": _data_url(image_bytes, mime)}},
    ])
    resp = llm.invoke([message])
    return strip_think(resp.content)
