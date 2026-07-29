from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Supported campaign intents. Drives channel prompt framing (e.g. email.j2).
# "general" is the backward-compatible default for briefs that omit the field.
CampaignType = Literal[
    "general",
    "promotional",
    "product_launch",
    "newsletter",
    "re_engagement",
    "announcement",
    "seasonal",
]

CAMPAIGN_TYPES: tuple[str, ...] = (
    "general",
    "promotional",
    "product_launch",
    "newsletter",
    "re_engagement",
    "announcement",
    "seasonal",
)


class BriefPayload(BaseModel):
    brand: str = Field(..., min_length=1, max_length=100, description="Brand name")
    channel: Literal["email", "linkedin", "social", "ad", "blog"] = Field(
        ..., description="Target marketing channel"
    )
    persona: str = Field(..., min_length=1, max_length=100, description="Target persona name")
    key_message: str = Field(
        ..., min_length=10, max_length=1000, description="Core message the content must convey"
    )
    campaign_type: CampaignType = Field(
        default="general",
        description="Campaign intent — shapes prompt framing (e.g. product_launch, newsletter, re_engagement).",
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="Optional extra constraints (e.g. required phrases, allowed domains)"
    )
    # [image-based-campaign] Per-brief switch. False returns text-only copy and
    # skips image generation entirely — no provider call, no storage write.
    generate_image: bool = Field(
        default=True, description="Generate a campaign image alongside the copy"
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
            "campaign_type": "product_launch",
            "constraints": {},
        }
    }}


class BriefResponse(BaseModel):
    brief_id: str
    status: str = "pending"
    message: str = "Brief accepted. Pipeline started."
