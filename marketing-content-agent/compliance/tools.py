from __future__ import annotations

import re
from typing import Any

# ─── Restricted word lists per channel ───────────────────────────────────────

_GLOBAL_RESTRICTED: list[str] = [
    "guaranteed", "free money", "no risk", "100% safe",
    "miracle", "instant results", "secret", "limited time offer",
    "act now", "you're a winner", "click here",
]

_CHANNEL_RESTRICTED: dict[str, list[str]] = {
    "email": ["unsubscribe bait", "phishing", "spam"],
    "linkedin": ["get rich quick", "pyramid", "mlm"],
    "social": ["nsfw", "offensive", "hate"],
    "ad": ["false claim", "misleading", "deceptive"],
    "blog": ["plagiarised", "duplicate content"],
}

# ─── Character limits per channel ────────────────────────────────────────────

_CHAR_LIMITS: dict[str, int] = {
    "email": 2000,
    "linkedin": 3000,
    "social": 500,
    "ad": 300,
    "blog": 10000,
}

# Fields measured against persona/channel limits (rest is labels or extras).
_LIMIT_FIELDS: dict[str, tuple[str, ...]] = {
    "email": ("body", "content", "copy"),
    "ad": ("body", "copy"),
    "social": ("copy", "body", "content"),
    "linkedin": ("hook", "bullets", "cta"),
    "blog": ("sections", "title", "meta_description"),
}


# Platform-specific caps for fields excluded from persona body/copy limits.
_FIELD_CAPS: dict[str, dict[str, int]] = {
    "email": {"subject": 60, "cta": 30},
    "ad": {"headline": 30, "cta": 15},
}


def length_from_metadata(channel: str, draft_metadata: dict[str, Any] | None) -> int | None:
    """Return limit-relevant character count from structured draft fields."""
    return _length_from_metadata(channel, draft_metadata)


def _trim_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    for sep in (". ", " ", ""):
        idx = cut.rfind(sep) if sep else -1
        if idx > max_len * 0.6:
            return cut[: idx + (1 if sep == ". " else 0)].rstrip()
    return cut.rstrip()


def clamp_draft_metadata(
    data: dict[str, Any],
    channel: str,
    char_limit: int,
) -> dict[str, Any]:
    """Trim structured draft fields so limit-relevant content fits persona/channel caps."""
    clamped = dict(data)
    ch = channel.lower()

    for field, cap in _FIELD_CAPS.get(ch, {}).items():
        if clamped.get(field):
            clamped[field] = _trim_text(str(clamped[field]), cap)

    if ch in {"email", "ad", "social"}:
        for key in _LIMIT_FIELDS.get(ch, ()):
            if clamped.get(key):
                clamped[key] = _trim_text(str(clamped[key]), char_limit)
                break
        return clamped

    if ch == "linkedin":
        while True:
            current = _length_from_metadata(ch, clamped)
            if current is None or current <= char_limit:
                break
            over = current - char_limit
            if clamped.get("bullets"):
                bullets = list(clamped["bullets"])
                idx = max(range(len(bullets)), key=lambda i: len(str(bullets[i])))
                bullets[idx] = _trim_text(str(bullets[idx]), max(20, len(str(bullets[idx])) - over))
                clamped["bullets"] = bullets
                continue
            if clamped.get("hook") and len(str(clamped["hook"])) > 40:
                clamped["hook"] = _trim_text(str(clamped["hook"]), len(str(clamped["hook"])) - over)
                continue
            if clamped.get("cta"):
                clamped["cta"] = _trim_text(str(clamped["cta"]), max(20, len(str(clamped["cta"])) - over))
                continue
            break
        return clamped

    if ch == "blog":
        while True:
            current = _length_from_metadata(ch, clamped)
            if current is None or current <= char_limit:
                break
            sections = list(clamped.get("sections") or [])
            if sections:
                last = dict(sections[-1])
                content = str(last.get("content") or "")
                if content:
                    last["content"] = _trim_text(content, max(100, len(content) - (current - char_limit)))
                    sections[-1] = last
                    clamped["sections"] = sections
                    continue
            if clamped.get("meta_description"):
                meta = str(clamped["meta_description"])
                clamped["meta_description"] = _trim_text(meta, max(80, len(meta) - (current - char_limit)))
                continue
            if clamped.get("title"):
                title = str(clamped["title"])
                clamped["title"] = _trim_text(title, max(40, len(title) - (current - char_limit)))
                continue
            break
        return clamped

    return clamped


def _length_from_metadata(channel: str, draft_metadata: dict[str, Any] | None) -> int | None:
    """Return limit-relevant character count from structured draft fields."""
    if not draft_metadata:
        return None

    ch = channel.lower()
    fields = _LIMIT_FIELDS.get(ch)
    if not fields:
        return None

    if ch == "linkedin":
        total = 0
        if draft_metadata.get("hook"):
            total += len(str(draft_metadata["hook"]))
        for bullet in draft_metadata.get("bullets") or []:
            total += len(str(bullet))
        if draft_metadata.get("cta"):
            total += len(str(draft_metadata["cta"]))
        return total if total else None

    if ch == "blog":
        total = 0
        if draft_metadata.get("title"):
            total += len(str(draft_metadata["title"]))
        for section in draft_metadata.get("sections") or []:
            if isinstance(section, dict):
                if section.get("subheading"):
                    total += len(str(section["subheading"]))
                if section.get("content"):
                    total += len(str(section["content"]))
        if draft_metadata.get("meta_description"):
            total += len(str(draft_metadata["meta_description"]))
        return total if total else None

    for key in fields:
        value = draft_metadata.get(key)
        if value:
            return len(str(value))
    return None


def _effective_length(
    draft: str,
    channel: str,
    draft_metadata: dict[str, Any] | None = None,
) -> int:
    """Character count used for limit checks (body/copy only where applicable)."""
    meta = draft_metadata or {}
    meta_len = _length_from_metadata(channel, meta if meta else None)
    if meta_len is not None:
        return meta_len

    # Fallback: parse JSON-ish drafts when metadata was not stored
    if draft.lstrip().startswith(("{", "```")) or '"body"' in draft or '"subject"' in draft:
        try:
            from graph.draft_parser import parse_llm_draft

            parsed = parse_llm_draft(draft, channel)
            if parsed:
                parsed_len = _length_from_metadata(channel, parsed)
                if parsed_len is not None:
                    return parsed_len
        except Exception:
            pass

    return len(draft)

# ─── Required phrases per channel ────────────────────────────────────────────

_REQUIRED_PHRASES: dict[str, list[str]] = {
    "email": [],
    "linkedin": [],
    "social": [],
    "ad": [],
    "blog": [],
}

# ─── URL format pattern ───────────────────────────────────────────────────────

_URL_PATTERN = re.compile(
    r"https?://"
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
    r"localhost|\d{1,3}(?:\.\d{1,3}){3})"
    r"(?::\d+)?"
    r"(?:/?|[/?]\S+)",
    re.IGNORECASE,
)

_SUSPICIOUS_URL_PATTERN = re.compile(
    r"https?://(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly)/\S+",
    re.IGNORECASE,
)


def check_restricted_words(draft: str, channel: str, brand: str = "") -> list[str]:
    """
    Check for globally and channel-specifically restricted words/phrases.
    Returns a list of violation strings (empty = pass).
    """
    violations: list[str] = []
    lower = draft.lower()

    for word in _GLOBAL_RESTRICTED:
        if word.lower() in lower:
            violations.append(f"Restricted word/phrase found: '{word}'")

    channel_words = _CHANNEL_RESTRICTED.get(channel.lower(), [])
    for word in channel_words:
        if word.lower() in lower:
            violations.append(f"Channel-restricted phrase for {channel}: '{word}'")

    if brand:
        brand_lower = brand.lower()
        competitor_mentions = [w for w in lower.split() if w != brand_lower and len(w) > 3]
        forbidden_comparisons = ["better than", "unlike", "vs.", "versus", "compared to"]
        for fc in forbidden_comparisons:
            if fc in lower:
                violations.append(f"Comparative claim detected: '{fc}' — review for false advertising")

    return violations


def check_char_limits(
    draft: str,
    channel: str,
    persona_limits: dict[str, int] | None = None,
    draft_metadata: dict[str, Any] | None = None,
) -> list[str]:
    """
    Verify the draft length does not exceed the channel + persona char limits.
    Counts only limit-relevant fields per channel:
      - email: body only (not subject/CTA labels)
      - linkedin: hook + bullets + cta (not hashtags)
      - social: copy only (not hashtags)
      - ad: body only (not headline/CTA labels)
      - blog: title + sections + meta (full article text)
    Returns a list of violation strings (empty = pass).
    """
    violations: list[str] = []
    length = _effective_length(draft, channel, draft_metadata)

    channel_limit = _CHAR_LIMITS.get(channel.lower())
    if channel_limit and length > channel_limit:
        violations.append(
            f"Draft exceeds {channel} channel limit: {length} chars > {channel_limit} chars"
        )

    if persona_limits:
        persona_key = f"char_limit_{channel.lower()}"
        persona_limit = persona_limits.get(persona_key)
        if persona_limit and length > persona_limit:
            violations.append(
                f"Draft exceeds persona limit for {channel}: {length} chars > {persona_limit} chars"
            )

    return violations


def check_required_phrases(
    draft: str, channel: str, extra_required: list[str] | None = None
) -> list[str]:
    """
    Ensure all required phrases for the channel (and any extra ones) are present.
    Returns a list of violation strings (empty = pass).
    """
    violations: list[str] = []
    lower = draft.lower()

    required = list(_REQUIRED_PHRASES.get(channel.lower(), []))
    if extra_required:
        required.extend(extra_required)

    for phrase in required:
        if phrase.lower() not in lower:
            violations.append(f"Required phrase missing: '{phrase}'")

    return violations


def check_url_format(draft: str, channel: str, allowed_domains: list[str] | None = None) -> list[str]:
    """
    Validate any URLs present in the draft:
      - Must match the URL regex.
      - Must not be link-shortened (suspicious).
      - Must be in the allowed_domains list if provided.
    Returns a list of violation strings (empty = pass).
    """
    violations: list[str] = []
    urls = _URL_PATTERN.findall(draft)

    for url in urls:
        if _SUSPICIOUS_URL_PATTERN.match(url):
            violations.append(f"Shortened/suspicious URL detected: '{url}'")
            continue

        if allowed_domains:
            domain_match = any(
                domain.lower() in url.lower() for domain in allowed_domains
            )
            if not domain_match:
                violations.append(
                    f"URL domain not in allowed list: '{url}' "
                    f"(allowed: {', '.join(allowed_domains)})"
                )

    return violations


def run_all_checks(
    draft: str,
    channel: str,
    brand: str = "",
    persona_limits: dict[str, int] | None = None,
    extra_required: list[str] | None = None,
    allowed_domains: list[str] | None = None,
    draft_metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Convenience function: run all four checks and aggregate violations."""
    violations: list[str] = []
    violations.extend(check_restricted_words(draft, channel, brand))
    violations.extend(
        check_char_limits(draft, channel, persona_limits, draft_metadata=draft_metadata)
    )
    violations.extend(check_required_phrases(draft, channel, extra_required))
    violations.extend(check_url_format(draft, channel, allowed_domains))
    return violations
