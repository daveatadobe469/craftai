from graph.draft_parser import compose_draft_text, parse_and_format_draft, parse_llm_draft

SAMPLE_EMAIL = '''```json
{
  "subject": "Unlock the Art of Fine Wine Pairing",
  "body": "Dear Connoisseur,

As an empty-nester with refined taste, you understand the joy of sharing exceptional experiences with loved ones. That's why we're excited to introduce our curated wine pairing collection, carefully crafted to elevate your gatherings.

Our expert sommeliers have handpicked rare vintages from renowned vineyards, ensuring each bottle is a masterclass in flavor and craftsmanship.

Warm regards,
[Your Name]",
  "cta": "Shop Wine Pairings"
}
```

This draft meets the requirements by:

* Exceeding the persona limit for email body
'''


def test_parse_email_with_literal_newlines_in_body():
    data = parse_llm_draft(SAMPLE_EMAIL, "email")
    assert data is not None
    assert data.get("subject") == "Unlock the Art of Fine Wine Pairing"
    assert "Dear Connoisseur" in data.get("body", "")
    assert data.get("cta") == "Shop Wine Pairings"


def test_format_email_not_raw_json():
    text, meta = parse_and_format_draft(SAMPLE_EMAIL, "email", 850)
    assert "```json" not in text
    assert "This draft meets" not in text
    assert text.startswith("Subject:")
    assert "Dear Connoisseur" in text
    assert meta.get("body")
    assert len(meta["body"]) <= 850


def test_compose_email_readable():
    data = parse_llm_draft(SAMPLE_EMAIL, "email")
    assert data
    out = compose_draft_text(data, "email")
    assert "Subject:" in out
    assert "CTA: Shop Wine Pairings" in out
