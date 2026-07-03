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
    draft: str, channel: str, persona_limits: dict[str, int] | None = None
) -> list[str]:
    """
    Verify the draft length does not exceed the channel + persona char limits.
    Returns a list of violation strings (empty = pass).
    """
    violations: list[str] = []
    length = len(draft)

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
) -> list[str]:
    """Convenience function: run all four checks and aggregate violations."""
    violations: list[str] = []
    violations.extend(check_restricted_words(draft, channel, brand))
    violations.extend(check_char_limits(draft, channel, persona_limits))
    violations.extend(check_required_phrases(draft, channel, extra_required))
    violations.extend(check_url_format(draft, channel, allowed_domains))
    return violations
