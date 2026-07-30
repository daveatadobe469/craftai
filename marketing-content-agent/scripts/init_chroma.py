#!/usr/bin/env python3
"""
init_chroma.py — Initialise all ChromaDB collections and seed with example data.
Run this ONCE before starting the application: `python scripts/init_chroma.py`
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from faker import Faker

from config import settings
from db.sqlite import create_tables
from rag import chroma_client as cc
from rag import embedder

fake = Faker()
fake.seed_instance(42)

_CHANNEL_EXAMPLES: list[dict[str, str]] = [
    {
        "channel": "email",
        "brand": "GlowBrand",
        "persona": "Premium Empty Nesters",
        "content": (
            "Subject: Your Exclusive Invitation — Platinum Skincare Collection\n"
            "Body: Dear {{first_name}}, experience the pinnacle of skincare innovation. "
            "Our platinum collection is clinically proven to reduce fine lines by 47% in 14 days. "
            "Formulated with rare marine peptides and gold-infused hyaluronic acid, "
            "this is luxury you can feel from the very first application.\n"
            "CTA: Discover the Collection"
        ),
    },
    {
        "channel": "email",
        "brand": "TechNova",
        "persona": "Digital Professionals",
        "content": (
            "Subject: Your Workflow Just Got 3× Faster\n"
            "Body: Hi {{first_name}}, TechNova Pro is now powered by our next-gen AI engine. "
            "Benchmark tests show a 312% throughput increase vs. legacy tools. "
            "Seamlessly integrates with Slack, Notion, and Jira in under 5 minutes.\n"
            "CTA: Start Free Trial"
        ),
    },
    {
        "channel": "linkedin",
        "brand": "TechNova",
        "persona": "Digital Professionals",
        "content": (
            "🚀 The future of productivity is here.\n\n"
            "We benchmarked 50+ enterprise tools. TechNova Pro came out on top — "
            "3× faster, 60% lower error rate, and a setup time under 5 minutes.\n\n"
            "Key takeaways:\n"
            "• AI-powered automation cuts manual tasks by 80%\n"
            "• Real-time collaboration for distributed teams\n"
            "• SOC-2 compliant from day one\n\n"
            "Ready to elevate your workflow? Link in bio.\n"
            "#Productivity #FutureOfWork #AI #TechNova"
        ),
    },
    {
        "channel": "social",
        "brand": "GlowBrand",
        "persona": "Smart-Saving Young Families",
        "content": (
            "Glow without breaking the bank ✨ "
            "Our bestselling vitamin C serum is now 30% off this weekend only. "
            "Real results, real savings. #GlowBrand #SkincareSale #AffordableBeauty"
        ),
    },
    {
        "channel": "social",
        "brand": "FreshFarms",
        "persona": "Value Families",
        "content": (
            "Sunday farmers market energy 🥕🌿 "
            "Fresh, locally sourced, seasonal produce delivered to your door. "
            "Feed your family with food you can trust. #FreshFarms #FarmToTable #FamilyFirst"
        ),
    },
    {
        "channel": "ad",
        "brand": "GlowBrand",
        "persona": "Premium Empty Nesters",
        "content": (
            "Headline: Clinically Proven Glow\n"
            "Body: Dermatologist-approved platinum skincare. 47% fine line reduction in 14 days.\n"
            "CTA: Shop Now"
        ),
    },
    {
        "channel": "ad",
        "brand": "TechNova",
        "persona": "Digital Professionals",
        "content": (
            "Headline: Work Smarter Today\n"
            "Body: AI-powered productivity suite trusted by 50,000 professionals. Free 14-day trial.\n"
            "CTA: Try Free"
        ),
    },
    {
        "channel": "blog",
        "brand": "GlowBrand",
        "persona": "Premium Empty Nesters",
        "content": (
            "Title: The Science Behind Platinum Skincare — What the Research Shows\n\n"
            "Introduction: In a crowded skincare market, separating science from marketing "
            "can be challenging. GlowBrand's platinum collection is built on three decades of "
            "dermatological research…\n\n"
            "Key Ingredients: Marine peptides stimulate collagen synthesis. "
            "Gold-infused hyaluronic acid provides 72-hour hydration…\n\n"
            "Clinical Results: In our 12-week double-blind study with 200 participants, "
            "94% reported visibly smoother skin…\n\n"
            "Meta: Discover the science behind GlowBrand's platinum skincare — "
            "clinically proven ingredients for transformative results."
        ),
    },
    {
        "channel": "blog",
        "brand": "FreshFarms",
        "persona": "Value Families",
        "content": (
            "Title: 7 Ways Seasonal Eating Benefits Your Family's Health\n\n"
            "Introduction: Choosing seasonal, locally sourced produce isn't just "
            "an environmental choice — it's one of the smartest decisions for your family's nutrition.\n\n"
            "Nutritional Peak: Produce harvested at peak ripeness contains up to 3× more "
            "vitamins than out-of-season alternatives shipped thousands of miles.\n\n"
            "Budget Benefits: Seasonal produce costs 20–40% less than off-season imports "
            "at your local farmers market.\n\n"
            "Meta: Learn how seasonal eating from FreshFarms improves nutrition, "
            "saves money, and supports your local community."
        ),
    },
    # ── Per-campaign-type email exemplars (grounded, with compliance footer) ──
    {
        "channel": "email",
        "brand": "GlowBrand",
        "persona": "Premium Empty Nesters",
        "campaign_type": "newsletter",
        "content": (
            "Subject: This Month in Skin Science — 3 Reads for You\n"
            "Preview: Fresh research, a ritual refresh, and a member spotlight.\n"
            "Body: Hi {{first_name}}, here's what caught our dermatologists' attention this month. "
            "First, new data on marine peptides and collagen synthesis. Second, how to adapt your "
            "evening ritual for drier seasons. Third, a member who transformed her routine in 14 days.\n"
            "CTA: Read the Digest\n"
            "GlowBrand · Unsubscribe"
        ),
    },
    {
        "channel": "email",
        "brand": "GlowBrand",
        "persona": "Premium Empty Nesters",
        "campaign_type": "re_engagement",
        "content": (
            "Subject: We Saved Your Spot, {{first_name}}\n"
            "Preview: Your personalised ritual is still waiting.\n"
            "Body: It's been a while, and your skin's needs may have changed. Your saved Platinum "
            "regimen is still here — clinically shown to reduce fine lines by 47% in 14 days. "
            "Pick up right where you left off, no pressure.\n"
            "CTA: Resume My Routine\n"
            "GlowBrand · Unsubscribe"
        ),
    },
    {
        "channel": "email",
        "brand": "TechNova",
        "persona": "Digital Professionals",
        "campaign_type": "announcement",
        "content": (
            "Subject: Introducing TechNova Pro 2.0 — Faster, Together\n"
            "Preview: The update your workflow has been waiting for.\n"
            "Body: Hi {{first_name}}, TechNova Pro 2.0 is here. Real-time collaboration is now built "
            "in, benchmark throughput is up 312% over the prior release, and setup still takes under "
            "5 minutes. Here's what's new and why it matters for your team.\n"
            "CTA: See What's New\n"
            "TechNova · Unsubscribe"
        ),
    },
]

_SOCIAL_EXAMPLES: list[dict[str, str]] = [
    {
        "platform": "twitter",
        "brand": "GlowBrand",
        "content": "Your skin deserves science, not shortcuts. ✨ #GlowBrand #CleanBeauty",
    },
    {
        "platform": "instagram",
        "brand": "GlowBrand",
        "content": "Morning ritual ☀️ Our platinum serum + SPF = the only duo you need. #GlowUp #GlowBrand",
    },
    {
        "platform": "twitter",
        "brand": "TechNova",
        "content": "3 hours saved. 0 context switches. TechNova Pro launches Monday. #Productivity #AI",
    },
    {
        "platform": "instagram",
        "brand": "FreshFarms",
        "content": "From our farm to your table 🌿 Zero compromises, all flavour. #FreshFarms #EatClean",
    },
    {
        "platform": "twitter",
        "brand": "FreshFarms",
        "content": "Seasonal eating = peak nutrition + peak savings. Shop your local FreshFarms box. 🥦",
    },
]


def seed_campaigns() -> None:
    print("Seeding approved_campaigns collection…")
    texts = [e["content"] for e in _CHANNEL_EXAMPLES]
    embeddings = embedder.encode(texts)
    ids = [str(uuid.uuid4()) for _ in _CHANNEL_EXAMPLES]
    metadatas = [
        {
            "channel": e["channel"],
            "brand": e["brand"],
            "persona": e["persona"],
            "campaign_type": e.get("campaign_type", "general"),
            "type": "approved_campaign",
            "judge_score": 0.9,
        }
        for e in _CHANNEL_EXAMPLES
    ]
    cc.upsert_documents(cc.APPROVED_CAMPAIGNS, ids, embeddings, texts, metadatas)
    count = cc.collection_count(cc.APPROVED_CAMPAIGNS)
    print(f"✓ approved_campaigns: {count} document(s)")


def seed_social_content() -> None:
    print("Seeding social_content collection…")
    texts = [e["content"] for e in _SOCIAL_EXAMPLES]
    embeddings = embedder.encode(texts)
    ids = [str(uuid.uuid4()) for _ in _SOCIAL_EXAMPLES]
    metadatas = [
        {
            "platform": e["platform"],
            "brand": e["brand"],
            "type": "social_content",
        }
        for e in _SOCIAL_EXAMPLES
    ]
    cc.upsert_documents(cc.SOCIAL_CONTENT, ids, embeddings, texts, metadatas)
    count = cc.collection_count(cc.SOCIAL_CONTENT)
    print(f"✓ social_content: {count} document(s)")


def main() -> None:
    print(f"ChromaDB persist dir: {settings.CHROMA_PERSIST_DIR}")
    cc.init_collections()
    create_tables()
    seed_campaigns()
    seed_social_content()
    print("\n✅ ChromaDB initialisation complete.")


if __name__ == "__main__":
    main()
