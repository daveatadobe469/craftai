from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from db.sqlite import get_personas

router = APIRouter()


class PersonaSummary(BaseModel):
    name: str
    description: str
    age_range: str
    income_bracket: str
    interests: list[str]
    pain_points: list[str]
    preferred_tone: str
    char_limit_email: int
    char_limit_social: int
    char_limit_linkedin: int
    char_limit_ad: int
    char_limit_blog: int
    persona_id: str | None = None
    profile: dict[str, object] = {}
    recommended_angles: list[str] = []
    channel_playbook: dict[str, object] = {}
    language_guardrails: dict[str, object] = {}
    voice_attributes: list[str] = []


class PersonasResponse(BaseModel):
    personas: list[PersonaSummary]


@router.get("/personas", response_model=PersonasResponse)
async def list_personas() -> PersonasResponse:
    records = get_personas()
    return PersonasResponse(personas=[PersonaSummary(**record) for record in records])
