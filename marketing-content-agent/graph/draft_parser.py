from __future__ import annotations

import json
import re
from typing import Any

from compliance.tools import clamp_draft_metadata


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _strip_trailing_commentary(text: str) -> str:
    """Remove LLM explanation after the JSON object (e.g. 'This draft meets...')."""
    if not text.startswith("{"):
        idx = text.find("{")
        if idx >= 0:
            text = text[idx:]
    slice_ = _balanced_json_slice(text)
    return slice_ or text


def _balanced_json_slice(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _fix_unescaped_newlines(blob: str) -> str:
    """Turn literal newlines inside JSON strings into \\n (common LLM mistake)."""
    out: list[str] = []
    in_string = False
    escape = False
    for ch in blob:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch == "\n":
            out.append("\\n")
            continue
        if in_string and ch in "\r\t":
            out.append(" " if ch == "\r" else " ")
            continue
        out.append(ch)
    return "".join(out)


def _try_load_json(blob: str) -> dict[str, Any] | None:
    for candidate in (blob, _fix_unescaped_newlines(blob)):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def _extract_email_fields(raw: str) -> dict[str, Any] | None:
    subject = re.search(r'"subject"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    body = re.search(r'"body"\s*:\s*"(.*?)"\s*,\s*"cta"\s*:', raw, re.DOTALL)
    cta = re.search(r'"cta"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    if not body:
        body = re.search(r'"body"\s*:\s*"(.*?)"\s*\n?\s*\}', raw, re.DOTALL)
    if not body:
        return None
    data: dict[str, Any] = {"body": body.group(1).replace("\\n", "\n").strip()}
    if subject:
        data["subject"] = subject.group(1).replace("\\n", " ").strip()
    if cta:
        data["cta"] = cta.group(1).replace("\\n", " ").strip()
    return data


def _extract_by_channel(raw: str, channel: str) -> dict[str, Any] | None:
    ch = channel.lower()
    if ch == "email":
        return _extract_email_fields(raw)
    return None


def parse_llm_draft(raw: str, channel: str) -> dict[str, Any] | None:
    """Parse structured draft JSON from an LLM response."""
    cleaned = _strip_fences(raw)
    cleaned = _strip_trailing_commentary(cleaned)
    blob = _balanced_json_slice(cleaned) or cleaned

    data = _try_load_json(blob)
    if data:
        return data

    return _extract_by_channel(blob, channel) or _extract_by_channel(raw, channel)


def compose_draft_text(data: dict[str, Any], channel: str) -> str:
    """Flatten structured draft JSON into readable marketing copy."""
    ch = channel.lower()
    parts: list[str] = []

    if ch == "email":
        if data.get("subject"):
            parts.append(f"Subject: {data['subject']}")
        body = data.get("body") or data.get("content") or data.get("copy")
        if body:
            parts.append(str(body))
        if data.get("cta"):
            parts.append(f"CTA: {data['cta']}")
        return "\n\n".join(parts)

    if ch == "linkedin":
        if data.get("hook"):
            parts.append(str(data["hook"]))
        for bullet in data.get("bullets") or []:
            if bullet:
                parts.append(f"• {bullet}")
        if data.get("cta"):
            parts.append(str(data["cta"]))
        tags = data.get("hashtags") or []
        if tags:
            parts.append(" ".join(str(t) for t in tags))
        return "\n\n".join(parts)

    if ch == "social":
        copy = data.get("copy") or data.get("body") or data.get("content")
        if copy:
            parts.append(str(copy))
        tags = data.get("hashtags") or []
        if tags:
            parts.append(" ".join(str(t) for t in tags))
        return "\n\n".join(parts)

    if ch == "ad":
        if data.get("headline"):
            parts.append(f"Headline: {data['headline']}")
        body = data.get("body") or data.get("copy")
        if body:
            parts.append(str(body))
        if data.get("cta"):
            parts.append(f"CTA: {data['cta']}")
        return "\n\n".join(parts)

    if ch == "blog":
        if data.get("title"):
            parts.append(str(data["title"]))
        for section in data.get("sections") or []:
            if isinstance(section, dict):
                if section.get("subheading"):
                    parts.append(str(section["subheading"]))
                if section.get("content"):
                    parts.append(str(section["content"]))
        if data.get("meta_description"):
            parts.append(f"Meta: {data['meta_description']}")
        return "\n\n".join(parts)

    for key in ("body", "copy", "content"):
        if data.get(key):
            return str(data[key])
    return json.dumps(data, indent=2)


def parse_and_format_draft(
    raw: str, channel: str, char_limit: int
) -> tuple[str, dict[str, Any]]:
    """
    Parse LLM output, clamp to persona limits, return (display_text, metadata).
    Never returns raw JSON fences to the UI when parsing succeeds.
    """
    data = parse_llm_draft(raw, channel)
    if not data:
        # Last resort: strip fences/commentary for display only
        cleaned = _strip_fences(raw)
        slice_ = _balanced_json_slice(_strip_trailing_commentary(cleaned))
        if slice_:
            cleaned = slice_
        return cleaned.strip(), {}

    data = clamp_draft_metadata(data, channel, char_limit)
    return compose_draft_text(data, channel), data
