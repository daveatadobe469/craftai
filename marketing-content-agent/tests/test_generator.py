"""Tests for draft JSON parsing and composition."""

from compliance.tools import clamp_draft_metadata
from graph.draft_parser import compose_draft_text, parse_and_format_draft, parse_llm_draft


def test_parse_email_includes_subject_body_and_cta():
    raw = """{"subject": "Summer glow awaits", "body": "Discover our premium range.", "cta": "Shop Now"}"""
    draft, meta = parse_and_format_draft(raw, "email", 750)
    assert "Subject: Summer glow awaits" in draft
    assert "premium range" in draft
    assert "CTA: Shop Now" in draft
    assert meta["subject"] == "Summer glow awaits"


def test_clamp_trims_oversized_email_body():
    data = {"subject": "Hi", "body": "A" * 800, "cta": "Go"}
    clamped = clamp_draft_metadata(data, "email", 650)
    assert len(clamped["body"]) <= 650


def test_compose_linkedin_flattens_structure():
    data = {
        "hook": "Your skin deserves better.",
        "bullets": ["Point one", "Point two"],
        "cta": "Learn more today.",
        "hashtags": ["#skincare", "#glow"],
    }
    draft = compose_draft_text(data, "linkedin")
    assert "Your skin deserves better." in draft
    assert "• Point one" in draft
    assert "#skincare" in draft
