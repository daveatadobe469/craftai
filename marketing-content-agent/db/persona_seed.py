from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PERSONAS_DIR = Path(__file__).parent.parent / "data" / "personas"

_CHAR_LIMITS: dict[str, tuple[int, int, int, int, int]] = {
    # spend_tier -> email, social, linkedin, ad, blog
    "premium":       (850, 280, 1300, 200, 4000),
    "high":          (800, 270, 1200, 180, 3500),
    "moderate-high": (750, 280, 1100, 170, 3200),
    "moderate":      (750, 260, 1100, 160, 3200),
    "low":           (650, 240, 900, 140, 2500),
}

_PAIN_DEFAULTS: dict[str, list[str]] = {
    "premium":       ["quality concerns", "lack of exclusivity", "generic marketing"],
    "high":          ["unclear value", "inconsistent quality", "poor curation"],
    "moderate-high": ["time constraints", "decision fatigue", "unclear ROI"],
    "moderate":      ["budget trade-offs", "convenience gaps", "trust barriers"],
    "low":           ["tight budget", "price sensitivity", "unexpected expenses"],
}


def _tone(raw: str) -> str:
    return raw.split(";")[0].strip() if raw else "professional"


def _description(persona: dict[str, Any]) -> str:
    demo = persona.get("demographics") or {}
    behavior = persona.get("behavior") or {}
    cs = persona.get("content_strategy") or {}
    categories = ", ".join(behavior.get("top_spend_categories") or [])
    tone = cs.get("tone_preference", "")
    parts = [
        f"Age {demo.get('age_band', 'unknown')}, income {demo.get('income_band', 'unknown')}.",
        f"Spend tier: {behavior.get('spend_tier', 'moderate')}.",
    ]
    if categories:
        parts.append(f"Top categories: {categories}.")
    if tone:
        parts.append(tone)
    return " ".join(parts)


def _pain_points(persona: dict[str, Any]) -> list[str]:
    behavior = persona.get("behavior") or {}
    triggers = behavior.get("purchase_triggers")
    if isinstance(triggers, list) and triggers:
        return [str(t) for t in triggers[:5]]
    tier = str(behavior.get("spend_tier") or "moderate")
    return list(_PAIN_DEFAULTS.get(tier, _PAIN_DEFAULTS["moderate"]))


def _char_limits(persona: dict[str, Any]) -> dict[str, int]:
    behavior = persona.get("behavior") or {}
    tier = str(behavior.get("spend_tier") or "moderate")
    email, social, linkedin, ad, blog = _CHAR_LIMITS.get(tier, _CHAR_LIMITS["moderate"])
    # Young-professional digital native: tighter email subjects, standard social
    if persona.get("persona_id") == "P06":
        email, social, linkedin = 700, 280, 1000
    return {
        "char_limit_email": email,
        "char_limit_social": social,
        "char_limit_linkedin": linkedin,
        "char_limit_ad": ad,
        "char_limit_blog": blog,
    }


def persona_record_from_json(persona: dict[str, Any]) -> dict[str, Any]:
    """Map rich cluster persona JSON to SQLite personas row fields."""
    demo = persona.get("demographics") or {}
    cs = persona.get("content_strategy") or {}
    limits = _char_limits(persona)
    return {
        "persona_id": persona.get("persona_id", ""),
        "name": persona["name"],
        "description": _description(persona),
        "age_range": demo.get("age_band", "25-45"),
        "income_bracket": demo.get("income_band", "middle"),
        "interests": cs.get("recommended_angles") or [],
        "pain_points": _pain_points(persona),
        "preferred_tone": _tone(cs.get("tone_preference", "professional")),
        "profile_json": persona,
        **limits,
    }


def load_persona_files(directory: Path | None = None) -> list[dict[str, Any]]:
    """Load P*.json persona files (excludes personas_all.json)."""
    root = directory or _PERSONAS_DIR
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("P*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("name"):
            records.append(persona_record_from_json(data))
    return records


def load_personas_all(directory: Path | None = None) -> list[dict[str, Any]]:
    """Load personas from personas_all.json if individual files are missing."""
    root = directory or _PERSONAS_DIR
    all_path = root / "personas_all.json"
    if not all_path.exists():
        return []
    data = json.loads(all_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [persona_record_from_json(p) for p in data if p.get("name")]
