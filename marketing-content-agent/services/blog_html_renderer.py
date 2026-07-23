from __future__ import annotations

import html
import re
from typing import Any


def _esc(text: Any) -> str:
    return html.escape(str(text or ""), quote=True)


def _reading_time(sections: list[dict[str, Any]]) -> int:
    words = sum(len(str(s.get("content", "")).split()) for s in sections)
    return max(1, round(words / 200))


def _paragraphs(text: str) -> str:
    """Split section content into <p> blocks on blank lines / newlines."""
    chunks = [p.strip() for p in re.split(r"\n\s*\n|\n", text.strip()) if p.strip()]
    if not chunks:
        chunks = [text.strip()]
    return "\n".join(f"        <p>{_esc(chunk)}</p>" for chunk in chunks)


def render_blog_html(
    brand: str,
    persona: str,
    key_message: str,
    draft_metadata: dict[str, Any],
    fallback_draft_text: str = "",
) -> str:
    """
    Render a self-contained, responsive HTML blog page (embedded CSS, no
    external assets) from a generated blog draft's structured metadata:
      {"title": str, "sections": [{"subheading": str, "content": str}, ...],
       "meta_description": str}

    Falls back to a single section built from `fallback_draft_text` when the
    LLM output could not be parsed into structured title/sections (e.g. the
    model didn't return valid JSON) so a blog page can still be produced.
    """
    title = draft_metadata.get("title") or f"{brand}: {key_message[:60]}"
    sections: list[dict[str, Any]] = [
        s for s in (draft_metadata.get("sections") or []) if isinstance(s, dict)
    ]
    if not sections and fallback_draft_text.strip():
        sections = [{"subheading": "", "content": fallback_draft_text.strip()}]
    meta_description = draft_metadata.get("meta_description") or ""
    reading_time = _reading_time(sections)

    sections_html_parts: list[str] = []
    for i, section in enumerate(sections):
        subheading = section.get("subheading") or ""
        content = str(section.get("content") or "")
        heading_html = f"        <h2>{_esc(subheading)}</h2>\n" if subheading else ""

        sections_html_parts.append(
            f"""
      <section class="post-section">
{heading_html}{_paragraphs(content)}
      </section>"""
        )

        # Callout: pull the meta description in as a "key takeaway" box
        # right after the first section, where a reader has enough context
        # to appreciate the summary but hasn't lost interest yet.
        if i == 0 and meta_description:
            sections_html_parts.append(
                f"""
      <aside class="callout">
        <span class="callout-label">Key takeaway</span>
        <p>{_esc(meta_description)}</p>
      </aside>"""
            )

    sections_html = "\n".join(sections_html_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(title)}</title>
<meta name="description" content="{_esc(meta_description)}">
<style>
  :root {{
    --ink: #1a1d29;
    --ink-soft: #4a5068;
    --muted: #7b8299;
    --accent: #6d5bff;
    --accent-soft: #f0edff;
    --border: #e8e9f0;
    --bg: #ffffff;
    --max-w: 720px;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    line-height: 1.7;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{
    max-width: var(--max-w);
    margin: 0 auto;
    padding: 56px 24px 96px;
  }}
  .eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent);
    background: var(--accent-soft);
    padding: 6px 14px;
    border-radius: 999px;
    margin-bottom: 24px;
  }}
  h1 {{
    font-size: clamp(1.9rem, 4vw, 2.6rem);
    line-height: 1.2;
    letter-spacing: -0.02em;
    margin: 0 0 20px;
    color: var(--ink);
  }}
  .meta-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px 20px;
    font-size: 0.9rem;
    color: var(--muted);
    padding-bottom: 28px;
    margin-bottom: 32px;
    border-bottom: 1px solid var(--border);
  }}
  .meta-bar span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .post-section {{ margin-bottom: 40px; }}
  .post-section h2 {{
    font-size: 1.35rem;
    letter-spacing: -0.01em;
    margin: 0 0 14px;
    color: var(--ink);
  }}
  .post-section p {{
    margin: 0 0 16px;
    color: var(--ink-soft);
    font-size: 1.05rem;
  }}
  .post-section p:last-child {{ margin-bottom: 0; }}
  .callout {{
    background: var(--accent-soft);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 18px 22px;
    margin: 8px 0 40px;
  }}
  .callout-label {{
    display: block;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 8px;
  }}
  .callout p {{
    margin: 0;
    color: var(--ink);
    font-size: 1rem;
    font-style: italic;
  }}
  footer {{
    margin-top: 56px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
    font-size: 0.85rem;
    color: var(--muted);
    text-align: center;
  }}
  @media (max-width: 480px) {{
    .wrap {{ padding: 36px 18px 72px; }}
  }}
</style>
</head>
<body>
  <div class="wrap">
    <span class="eyebrow">{_esc(brand)}</span>
    <h1>{_esc(title)}</h1>
    <div class="meta-bar">
      <span>📖 {reading_time} min read</span>
      <span>🎯 Written for {_esc(persona)}</span>
    </div>
{sections_html}
    <footer>Published by {_esc(brand)}</footer>
  </div>
</body>
</html>
"""
