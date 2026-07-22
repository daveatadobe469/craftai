from __future__ import annotations

import pytest

from compliance.tools import (
    _effective_length,
    check_char_limits,
    check_required_phrases,
    check_restricted_words,
    check_url_format,
    length_from_metadata,
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

    def test_email_checks_body_only_not_subject_labels(self):
        composed = "Subject: Hello\n\n" + ("A" * 500) + "\n\nCTA: Shop"
        metadata = {"subject": "Hello", "body": "A" * 500, "cta": "Shop"}
        persona_limits = {"char_limit_email": 650}
        result = check_char_limits(
            composed,
            channel="email",
            persona_limits=persona_limits,
            draft_metadata=metadata,
        )
        assert result == []

    def test_email_json_draft_uses_body_only_length(self):
        body = "A" * 500
        raw = f'{{"subject": "Hi", "body": "{body}", "cta": "Go"}}'
        assert _effective_length(raw, "email", {}) == 500

    def test_email_body_over_limit_still_fails(self):
        metadata = {"body": "A" * 700}
        result = check_char_limits(
            "Subject: x\n\n" + ("A" * 700),
            channel="email",
            persona_limits={"char_limit_email": 650},
            draft_metadata=metadata,
        )
        assert any("650" in v for v in result)


class TestEffectiveLength:
    def test_effective_length_uses_email_body(self):
        assert _effective_length(
            "Subject: Hi\n\nlong body\n\nCTA: Go",
            "email",
            {"body": "short body"},
        ) == len("short body")

    def test_linkedin_excludes_hashtags_from_limit(self):
        meta = {
            "hook": "Hi",
            "bullets": ["One"],
            "cta": "Go",
            "hashtags": ["#" + ("x" * 500)],
        }
        composed = "Hi\n\n• One\n\nGo\n\n" + ("#" + "x" * 500)
        assert check_char_limits(
            composed,
            channel="linkedin",
            persona_limits={"char_limit_linkedin": 50},
            draft_metadata=meta,
        ) == []

    def test_social_excludes_hashtags_from_limit(self):
        meta = {"copy": "A" * 200, "hashtags": ["#b" * 50]}
        composed = ("A" * 200) + "\n\n" + ("#b" * 50)
        assert check_char_limits(
            composed,
            channel="social",
            persona_limits={"char_limit_social": 280},
            draft_metadata=meta,
        ) == []

    def test_ad_excludes_headline_from_body_limit(self):
        meta = {"headline": "H" * 30, "body": "B" * 150, "cta": "Buy"}
        composed = f"Headline: {'H' * 30}\n\n{'B' * 150}\n\nCTA: Buy"
        assert check_char_limits(
            composed,
            channel="ad",
            persona_limits={"char_limit_ad": 170},
            draft_metadata=meta,
        ) == []

    def test_blog_uses_structured_sections(self):
        meta = {
            "title": "Title",
            "sections": [{"subheading": "S", "content": "C" * 1000}],
            "meta_description": "Meta",
        }
        composed = "Title\n\nS\n\n" + ("C" * 1000) + "\n\nMeta: Meta"
        assert check_char_limits(
            composed,
            channel="blog",
            persona_limits={"char_limit_blog": 5000},
            draft_metadata=meta,
        ) == []


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
