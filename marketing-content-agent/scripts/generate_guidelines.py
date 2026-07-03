#!/usr/bin/env python3
"""
generate_guidelines.py — Seed the brand_guidelines ChromaDB collection
with synthetic brand guideline documents using Faker.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from faker import Faker

from config import settings
from rag import chroma_client as cc
from rag import embedder

fake = Faker()

_GUIDELINES: list[dict[str, str]] = [
    {
        "brand": "GlowBrand",
        "content": (
            "GlowBrand tone of voice: warm, empowering, and science-forward. "
            "Always reference clinical trials or dermatologist approvals. "
            "Avoid superlatives without evidence. Use inclusive language. "
            "Primary CTA verbs: Discover, Transform, Elevate."
        ),
    },
    {
        "brand": "GlowBrand",
        "content": (
            "GlowBrand visual and copy guidelines: Use the colour palette Ivory (#FFFDF7) "
            "and Rose Gold (#B76E79). Headline copy must be under 60 characters. "
            "Never use 'cheap', 'bargain', or 'deal' — position around 'value' and 'investment'."
        ),
    },
    {
        "brand": "GlowBrand",
        "content": (
            "GlowBrand channel-specific rules: Email subject lines must personalise with "
            "first name token {{first_name}}. LinkedIn posts must include 3–5 relevant hashtags. "
            "Social posts must not exceed 240 characters for Twitter/X compatibility. "
            "Ad headlines must be under 30 characters."
        ),
    },
    {
        "brand": "TechNova",
        "content": (
            "TechNova brand voice: bold, innovative, data-driven. "
            "Lead with statistics and performance benchmarks. "
            "Avoid jargon; simplify complex concepts for a non-technical audience. "
            "Approved power words: Breakthrough, Next-gen, Precision, Seamless."
        ),
    },
    {
        "brand": "TechNova",
        "content": (
            "TechNova compliance rules: Never claim 'best' or '#1' without citing a source. "
            "All comparative claims must reference an industry report dated within 2 years. "
            "Do not use competitor brand names in paid advertising copy."
        ),
    },
    {
        "brand": "FreshFarms",
        "content": (
            "FreshFarms tone: honest, earthy, community-focused. "
            "Highlight farm-to-table transparency. "
            "Use 'seasonal', 'locally sourced', 'artisan' where accurate. "
            "Avoid greenwashing — only use 'organic' if certified."
        ),
    },
    {
        "brand": "Generic",
        "content": (
            "General marketing compliance: All content must comply with FTC guidelines. "
            "Disclose paid partnerships with #ad or #sponsored. "
            "Health claims require medical professional review before publication. "
            "No countdown timers or artificial scarcity messaging."
        ),
    },
    {
        "brand": "Generic",
        "content": (
            "Email marketing best practices: Subject lines 40–60 characters perform best. "
            "Include plain-text alternative. "
            "Unsubscribe link must be present and functional. "
            "Sender name must match brand identity."
        ),
    },
]


def seed_guidelines() -> None:
    print("Seeding brand_guidelines collection…")
    cc.init_collections()

    texts = [g["content"] for g in _GUIDELINES]
    embeddings = embedder.encode(texts)
    ids = [str(uuid.uuid4()) for _ in _GUIDELINES]
    metadatas = [{"brand": g["brand"], "type": "brand_guideline"} for g in _GUIDELINES]

    cc.upsert_documents(
        collection_name=cc.BRAND_GUIDELINES,
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    count = cc.collection_count(cc.BRAND_GUIDELINES)
    print(f"✓ brand_guidelines collection now has {count} document(s).")


if __name__ == "__main__":
    seed_guidelines()
