#!/usr/bin/env python3
"""
seed_personas.py — Load generated persona JSON files into the SQLite `personas` table.

The clustering task emits analytical persona profiles (demographics, behavior, channel
shares) whose shape differs from the operational `personas` schema the generator prompt
consumes. This script transforms each `P0*.json` into the table columns and upserts it.

Run:  python scripts/seed_personas.py
Source folder is `settings.PERSONAS_DIR` (default ./data/personas).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from db.sqlite import create_tables, upsert_persona


def map_persona(j: dict) -> dict:
    """Map a generated persona JSON dict to `personas` table columns."""
    cs = j.get("content_strategy", {}) or {}
    beh = j.get("behavior", {}) or {}
    demo = j.get("demographics", {}) or {}

    tone = cs.get("tone_preference", "professional")
    # interests: spend categories + recommended angles so both reach the Jinja2 prompt
    interests = list(beh.get("top_spend_categories", []) or []) + list(
        cs.get("recommended_angles", []) or []
    )
    # pain_points: only some personas carry purchase_triggers
    pain_points = list(beh.get("purchase_triggers", []) or [])

    return {
        "name": j["name"],
        "description": tone,
        "age_range": demo.get("age_band", "25-45"),
        "income_bracket": demo.get("income_band", "middle"),
        "interests": interests,
        "pain_points": pain_points,
        "preferred_tone": re.split(r"[;,]", tone)[0].strip() or "professional",
        # char_limit_* omitted — DB defaults apply.
    }


def _persona_files(folder: Path) -> list[Path]:
    """Individual persona files only — skip summary/aggregate artifacts."""
    return sorted(
        p for p in folder.glob("P*.json")
        if p.name not in {"personas_all.json", "clustering_summary.json"}
    )


def main() -> None:
    folder = Path(settings.PERSONAS_DIR)
    print(f"Persona source dir: {folder.resolve()}")

    if not folder.exists():
        print(f"⚠  Folder not found: {folder}. Nothing to seed.")
        return

    create_tables()

    files = _persona_files(folder)
    if not files:
        print("⚠  No P*.json persona files found.")
        return

    seeded = 0
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            mapped = map_persona(data)
            upsert_persona(**mapped)
            print(f"[ok]   {mapped['name']}  (from {path.name})")
            seeded += 1
        except Exception as exc:  # noqa: BLE001 — keep seeding remaining files
            print(f"[fail] {path.name}: {exc}")

    print(f"\nSeeded/updated {seeded} persona(s) into {settings.SQLITE_DB_PATH}")


if __name__ == "__main__":
    main()
