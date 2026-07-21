# [image-based-campaign] Shared renderer for the generated image + its compliance
# verdict. Used by both the Create-Brief review panel and the Search-Brief page so
# the two stay in sync.
from __future__ import annotations

import streamlit as st


def _score_color(score: float) -> str:
    return "#39ff14" if score >= 0.7 else "#ff9500" if score >= 0.4 else "#ff4444"


def _render_verdict(metadata: dict) -> None:
    """Show the image-judge result, if the judge ran."""
    score = metadata.get("image_judge_score")
    evidence = metadata.get("image_judge_evidence") or ""
    issues = metadata.get("image_judge_issues") or []

    # Judge ran but could not be evaluated (rate limit / API error) — fail open.
    if score is None:
        if evidence:
            st.info(f"Image compliance not evaluated — {evidence}", icon="ℹ️")
        return

    score = float(score)
    color = _score_color(score)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:8px 0 6px;">'
        f'<span style="font-size:0.72rem;color:#8aa3b8;letter-spacing:1px;">IMAGE COMPLIANCE</span>'
        f'<span style="font-size:1.1rem;font-weight:800;color:{color};">{score:.2f}</span>'
        f'<span style="font-size:0.72rem;color:#64748b;">/ 1.00 · {len(issues)} issue(s)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    for issue in issues:
        st.warning(issue, icon="⚠️")
    if evidence:
        with st.expander("Judge reasoning", expanded=False):
            st.caption(evidence)


def render_generated_image(metadata: dict | None, heading: str = "🖼️ Generated Image") -> None:
    """Render the campaign image and its compliance verdict. No-op when absent."""
    metadata = metadata or {}
    image_url = metadata.get("image_url")
    if not image_url:
        return

    st.markdown(
        f'<div style="font-size:0.72rem;color:#00d4ff;letter-spacing:2px;'
        f'text-transform:uppercase;margin:12px 0 8px 0;">{heading}</div>',
        unsafe_allow_html=True,
    )
    st.image(image_url, use_container_width=True)
    _render_verdict(metadata)
