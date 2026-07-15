#!/usr/bin/env python3
"""Seed P01–P06 cluster personas into SQLite from data/personas/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.sqlite import create_tables, get_personas, seed_cluster_personas


def main() -> None:
    create_tables()
    count = seed_cluster_personas()
    personas = get_personas()
    print(f"Seeded/updated {count} persona file(s). Total in DB: {len(personas)}")
    for p in personas:
        pid = p.get("persona_id") or "—"
        print(f"  {pid}  {p['name']}")


if __name__ == "__main__":
    main()
