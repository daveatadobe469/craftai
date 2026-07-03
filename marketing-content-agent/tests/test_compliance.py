from __future__ import annotations

import pytest

from compliance.tools import (
    check_char_limits,
    check_required_phrases,
    check_restricted_words,
    check_url_format,
    run_all_checks,
)


# ─── check_restricted_words ───────────────────────────────────────────────────

class TestCheckRestrictedWords:
    def test_clean_draft_returns_no_violations(self):
        draft = "Discover our premium skincare range. Transform your routine today."
        result = check_restricted_words(draft, channel="email", brand="GlowBrand")
        assert result == []

    def test_global_restricted_word_detected(self):
        draft = "Guaranteed results in 7 days or your money back."
        result = check_restricted_words(draft, channel="email")
        assert any("guaranteed" in v.lower() for v in result)

    def test_channel_restricted_word_linkedin(self):
        draft = "Join our get rich quick scheme today!"
        result = check_restricted_words(draft, channel="linkedin")
        assert any("get rich quick" in v.lower() for v in result)

    def test_multiple_restricted_words_returns_multiple_violations(self):
        draft = "No risk. Instant results. Act now for your free money offer!"
        result = check_restricted_words(draft, channel="social")
        assert len(result) >= 3


# ─── check_char_limits ────────────────────────────────────────────────────────

class TestCheckCharLimits:
    def test_within_limit_returns_no_violations(self):
        draft = "A" * 100
        result = check_char_limits(draft, channel="social")
        assert result == []

    def test_exceeds_social_limit_returns_violation(self):
        draft = "A" * 600
        result = check_char_limits(draft, channel="social")
        assert any("social" in v.lower() for v in result)

    def test_exceeds_persona_limit_returns_violation(self):
        draft = "A" * 300
        persona_limits = {"char_limit_ad": 150}
        result = check_char_limits(draft, channel="ad", persona_limits=persona_limits)
        assert any("persona" in v.lower() for v in result)

    def test_within_persona_limit_returns_no_violations(self):
        draft = "A" * 100
        persona_limits = {"char_limit_ad": 150}
        result = check_char_limits(draft, channel="ad", persona_limits=persona_limits)
        assert result == []


# ─── check_required_phrases ───────────────────────────────────────────────────

class TestCheckRequiredPhrases:
    def test_no_required_phrases_for_email_by_default(self):
        draft = "Hello world"
        result = check_required_phrases(draft, channel="email")
        assert result == []

    def test_extra_required_phrase_present_passes(self):
        draft = "Shop now and discover the difference."
        result = check_required_phrases(draft, channel="email", extra_required=["shop now"])
        assert result == []

    def test_extra_required_phrase_missing_fails(self):
        draft = "Discover our range of premium products."
        result = check_required_phrases(draft, channel="email", extra_required=["call today"])
        assert any("call today" in v.lower() for v in result)


# ─── check_url_format ─────────────────────────────────────────────────────────

class TestCheckUrlFormat:
    def test_no_urls_passes(self):
        draft = "Visit our store for more details."
        result = check_url_format(draft, channel="email")
        assert result == []

    def test_allowed_domain_passes(self):
        draft = "Visit us at https://glowbrand.com/skincare for more."
        result = check_url_format(draft, channel="email", allowed_domains=["glowbrand.com"])
        assert result == []

    def test_disallowed_domain_fails(self):
        draft = "Check out https://competitor.com/offer for a deal."
        result = check_url_format(draft, channel="email", allowed_domains=["glowbrand.com"])
        assert any("competitor.com" in v for v in result)

    def test_shortened_url_fails(self):
        draft = "See our offer at https://bit.ly/abc123 now."
        result = check_url_format(draft, channel="email")
        assert any("shortened" in v.lower() or "bit.ly" in v.lower() for v in result)


# ─── run_all_checks integration ───────────────────────────────────────────────

class TestRunAllChecks:
    def test_clean_draft_passes_all_checks(self):
        draft = "Discover premium skincare at https://glowbrand.com today. Elevate your routine."
        result = run_all_checks(
            draft=draft,
            channel="email",
            brand="GlowBrand",
            allowed_domains=["glowbrand.com"],
        )
        assert result == []

    def test_problematic_draft_fails_multiple_checks(self):
        draft = "Guaranteed! No risk! Act now at https://bit.ly/xyz. " + "A" * 600
        result = run_all_checks(
            draft=draft,
            channel="social",
            brand="GlowBrand",
        )
        assert len(result) >= 3
