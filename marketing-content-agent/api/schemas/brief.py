from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class BriefPayload(BaseModel):
    brand: str = Field(..., min_length=1, max_length=100, description="Brand name")
    channel: Literal["email", "linkedin", "social", "ad", "blog"] = Field(
        ..., description="Target marketing channel"
    )
    persona: str = Field(..., min_length=1, max_length=100, description="Target persona name")
    key_message: str = Field(
        ..., min_length=10, max_length=1000, description="Core message the content must convey"
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="Optional extra constraints (e.g. required phrases, allowed domains)"
    )

    @model_validator(mode="after")
    def _strip_whitespace(self) -> BriefPayload:
        self.brand = self.brand.strip()
        self.persona = self.persona.strip()
        self.key_message = self.key_message.strip()
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "brand": "GlowBrand",
            "channel": "email",
            "persona": "Premium Buyer",
            "key_message": "Introducing our new platinum skincare line — clinically proven results in 14 days.",
            "constraints": {},
        }
    }}


class BriefResponse(BaseModel):
    brief_id: str
    status: str = "pending"
    message: str = "Brief accepted. Pipeline started."
