from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DecisionPayload(BaseModel):
    decision: Literal["approved", "edited", "rejected"] = Field(
        ..., description="Human reviewer decision"
    )
    edits: Optional[str] = Field(
        default=None,
        description="Edited draft content (required when decision is 'edited')",
    )
    reviewer: str = Field(
        default="human",
        max_length=100,
        description="Reviewer identifier",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "decision": "approved",
            "edits": None,
            "reviewer": "marketing-lead@brand.com",
        }
    }}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Semantic search query")
    collection: Literal["approved_campaigns", "social_content", "brand_guidelines"] = Field(
        default="approved_campaigns"
    )
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict = Field(default_factory=dict)


class SearchResult(BaseModel):
    document: str
    metadata: dict
    score: float


class SearchResponse(BaseModel):
    query: str
    collection: str
    results: list[SearchResult]
    total: int
