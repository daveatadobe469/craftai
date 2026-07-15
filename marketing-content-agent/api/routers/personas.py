from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from db.sqlite import get_personas

router = APIRouter()


class PersonaSummary(BaseModel):
    name: str
    description: str
    preferred_tone: str
    age_range: str
    income_bracket: str
    interests: list[str]


class PersonasResponse(BaseModel):
    total: int
    names: list[str]
    personas: list[PersonaSummary]


@router.get("/personas", response_model=PersonasResponse, tags=["Personas"])
async def list_personas() -> PersonasResponse:
    """Return all seeded personas (names + summary fields) for UI selection."""
    loop = asyncio.get_event_loop()
    rows: list[dict[str, Any]] = await loop.run_in_executor(None, get_personas)
    personas = [
        PersonaSummary(
            name=r["name"],
            description=r.get("description", ""),
            preferred_tone=r.get("preferred_tone", ""),
            age_range=r.get("age_range", ""),
            income_bracket=r.get("income_bracket", ""),
            interests=r.get("interests", []),
        )
        for r in rows
    ]
    return PersonasResponse(
        total=len(personas),
        names=[p.name for p in personas],
        personas=personas,
    )
