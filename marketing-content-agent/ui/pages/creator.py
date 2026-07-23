from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Literal

import httpx
import streamlit as st
import streamlit.components.v1 as components

from ui.components.image_review import render_generated_image  # [image-based-campaign]
from ui.components.progress import (
    inject_progress_css,
    run_with_progress,
    show_ring,
    steps_to_percent,
)

_PIPELINE_TIMEOUT_S = 180

# ── Persona options (loaded from API) ─────────────────────────────────────────
_FALLBACK_PERSONAS: list[str] = [
    "Premium Empty Nesters",
    "Smart-Saving Young Families",
    "Premium Professionals",
    "Value Families",
    "Connected Families",
    "Digital Professionals",
]

_FALLBACK_PERSONA_DISPLAY: dict[str, str] = {name: name for name in _FALLBACK_PERSONAS}


def _persona_display_label(persona: dict) -> str:
    """Selectbox label — friendly persona name only."""
    return (persona.get("name") or "Unknown persona").strip()


def _fetch_persona_options(api_base: str) -> tuple[list[str], dict[str, str]]:
    """Return persona names and name→display labels from the API."""
    displays: dict[str, str] = {}
    try:
        r = httpx.get(f"{api_base}/personas", timeout=5.0)
        r.raise_for_status()
        personas = r.json().get("personas") or []
        if personas:
            names: list[str] = []
            for p in personas:
                name = p.get("name", "")
                if not name:
                    continue
                names.append(name)
                displays[name] = _persona_display_label(p)
            if names:
                return names, displays
    except Exception:
        pass
    return _FALLBACK_PERSONAS, dict(_FALLBACK_PERSONA_DISPLAY)

# ── Step data model ────────────────────────────────────────────────────────────
Status = Literal["pending", "running", "done", "error", "waiting"]


@dataclass
class Step:
    icon:    str
    label:   str
    status:  Status       = "pending"
    elapsed: float | None = None
    detail:  str          = ""


def _fresh_steps() -> list[Step]:
    return [
        Step("📋", "Brief Received"),
        Step("🎯", "Orchestrator — Validate & Plan"),
        Step("🔍", "RAG Retrieval — Vector Search"),
        Step("✍️",  "LLM Draft Generation"),
        Step("🛡️", "Compliance Check"),
        Step("👤", "Human Gate — Awaiting Your Approval"),
        Step("📚", "Curator — Index to Knowledge Base"),
    ]


# ── Step card HTML (inline styles, no <style> blocks) ─────────────────────────
_COLORS: dict[str, dict[str, str]] = {
    "pending": dict(bg="#121826", border="#1e293b", text="#475569",
                    badge_bg="#1e293b", badge_fg="#475569"),
    "running": dict(bg="#0b1a2e", border="#00d4ff", text="#00d4ff",
                    badge_bg="#091d3a", badge_fg="#00d4ff"),
    "done":    dict(bg="#091a0f", border="#39ff14", text="#39ff14",
                    badge_bg="#0a1e10", badge_fg="#39ff14"),
    "error":   dict(bg="#1a0909", border="#ff4444", text="#ff4444",
                    badge_bg="#200a0a", badge_fg="#ff4444"),
    "waiting": dict(bg="#1a1205", border="#ff9500", text="#ff9500",
                    badge_bg="#1e1505", badge_fg="#ff9500"),
}
_BADGE: dict[str, str] = {
    "pending": "PENDING",
    "running": "RUNNING",
    "done":    "DONE",
    "error":   "ERROR",
    "waiting": "ACTION REQUIRED",
}
_BAR: dict[str, str] = {
    "pending": "0", "running": "55", "done": "100",
    "error": "100", "waiting": "75",
}


def _step_html(step: Step, idx: int, elapsed: float) -> str:
    c = _COLORS[step.status]
    if step.status == "done" and step.elapsed is not None:
        time_str = f"{step.elapsed:.0f}s"
    elif step.status in ("running", "waiting"):
        time_str = f"{elapsed:.0f}s"
    else:
        time_str = ""
    detail_html = (
        f'<div style="font-size:0.7rem;color:#64748b;margin-top:3px;">{step.detail}</div>'
        if step.detail else ""
    )
    return (
        f'<div style="background:{c["bg"]};border:1px solid {c["border"]};'
        f'border-left:4px solid {c["border"]};border-radius:9px;'
        f'padding:10px 14px;margin-bottom:5px;display:flex;align-items:center;gap:12px;">'
        f'<span style="font-size:1.2rem;width:26px;text-align:center;">{step.icon}</span>'
        f'<div style="flex:1;min-width:0;">'
        f'  <div style="font-size:0.84rem;font-weight:600;color:{c["text"]};">'
        f'    <span style="font-size:0.66rem;opacity:0.55;margin-right:8px;">STEP {idx}</span>'
        f'    {step.label}</div>'
        f'  {detail_html}'
        f'  <div style="margin-top:6px;height:2px;background:#1e293b;border-radius:2px;">'
        f'    <div style="width:{_BAR[step.status]}%;height:100%;background:{c["border"]};'
        f'    border-radius:2px;"></div></div>'
        f'</div>'
        f'<div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px;">'
        f'  <span style="background:{c["badge_bg"]};color:{c["badge_fg"]};'
        f'  border:1px solid {c["badge_fg"]};font-size:0.64rem;font-weight:700;'
        f'  letter-spacing:.6px;padding:2px 8px;border-radius:10px;">'
        f'  {_BADGE[step.status]}</span>'
        f'  <span style="font-size:0.68rem;color:#475569;">{time_str}</span>'
        f'</div></div>'
    )


def _render_all(
    phs: list,
    steps: list[Step],
    elapsed: float,
    progress_ph=None,
) -> None:
    for i, (ph, step) in enumerate(zip(phs, steps), 1):
        ph.markdown(_step_html(step, i, elapsed), unsafe_allow_html=True)
    if progress_ph is not None:
        pct = steps_to_percent(steps)
        show_ring(
            progress_ph,
            pct,
            "Running AI pipeline…",
            f"{pct}% complete · {elapsed:.0f}s elapsed",
            height=240,
        )


# ── SSE → step state machine ───────────────────────────────────────────────────
def _update_steps(steps: list[Step], message: str, now: float, start: float) -> None:
    lo      = message.lower()
    elapsed = now - start

    def activate(idx: int, detail: str = "") -> None:
        for j in range(idx):
            if steps[j].status == "running":
                steps[j].status  = "done"
                steps[j].elapsed = elapsed
        if steps[idx].status == "pending":
            steps[idx].status = "running"
            if detail:
                steps[idx].detail = detail

    def complete(idx: int, detail: str = "") -> None:
        if steps[idx].status in ("pending", "running"):
            steps[idx].status  = "done"
            steps[idx].elapsed = elapsed
            if detail:
                steps[idx].detail = detail

    def after_bracket(msg: str) -> str:
        p = msg.find("]")
        return msg[p + 1:].strip()[:65] if p != -1 else msg[:65]

    if steps[0].status == "pending":
        steps[0].status  = "done"
        steps[0].elapsed = 0.0

    if "[orchestrator]" in lo:
        activate(1)
        if any(k in lo for k in ("validated", "plan ready", "plan (")):
            complete(1, after_bracket(message))

    if "starting retrieval" in lo or "hyde" in lo:
        activate(2, "HyDE rewrite + ChromaDB vector search")
    if "retrieved" in lo and any(k in lo for k in ("campaign", "guideline")):
        complete(2, after_bracket(message))

    if "calling llm" in lo:
        activate(3, "Generating channel-specific draft…")
    if "draft generated" in lo or ("ragas" in lo and "[generator]" in lo):
        complete(3, after_bracket(message))

    if "[compliance]" in lo:
        activate(4)
    if "judge score" in lo or "pass:" in lo:
        complete(4, after_bracket(message))

    if "[curator]" in lo:
        activate(6)
    if "indexed" in lo and "[curator]" in lo:
        complete(6, after_bracket(message))


# ── Inline Review Panel ────────────────────────────────────────────────────────
def _render_review_panel(api_base: str, brief_id: str) -> None:
    """Full inline review card shown when the pipeline pauses at Human Gate."""

    # Fetch current draft + compliance data
    try:
        def _fetch_draft():
            r = httpx.get(f"{api_base}/status/{brief_id}", timeout=10.0)
            r.raise_for_status()
            return r.json()

        draft_data = run_with_progress(
            "Loading draft for review…",
            _fetch_draft,
            estimated_seconds=4,
        )
    except Exception as exc:
        st.error(f"Cannot reach API: {exc}")
        return

    draft         = draft_data.get("draft", "")
    compliance    = draft_data.get("compliance_pass", False)
    judge_score   = float(draft_data.get("judge_score", 0.0))
    violations    = draft_data.get("rule_violations", [])
    channel       = draft_data.get("channel", "").upper()
    brand         = draft_data.get("brand", "")
    persona       = draft_data.get("persona", "")
    ragas_scores  = draft_data.get("ragas_scores", {})

    # Already decided — don't show again
    if draft_data.get("human_decision"):
        decision = draft_data["human_decision"]
        _color   = {"approved": "#39ff14", "edited": "#ff9500", "rejected": "#ff4444"}.get(decision, "#00d4ff")
        st.markdown(
            f'<div style="background:#0b1525;border:1px solid {_color};border-left:4px solid {_color};'
            f'border-radius:10px;padding:14px 18px;margin-bottom:16px;">'
            f'<span style="color:{_color};font-weight:700;font-size:0.9rem;">'
            f'Decision already submitted: {decision.upper()}</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear & Submit New Brief", type="secondary"):
            for k in ("gate_brief_id", "gate_decision_made", "active_brief_id"):
                st.session_state.pop(k, None)
            st.rerun()
        return

    # ── Header banner ─────────────────────────────────────────────────────────
    score_color  = "#39ff14" if judge_score >= 0.7 else "#ff9500" if judge_score >= 0.4 else "#ff4444"
    comp_color   = "#39ff14" if compliance else "#ff4444"
    comp_label   = "PASS" if compliance else "FAIL"

    st.markdown(
        f'<div style="background:linear-gradient(120deg,#1a1200 0%,#1e1505 60%,#120e02 100%);'
        f'border:1px solid #ff9500;border-left:4px solid #ff9500;border-radius:12px;'
        f'padding:16px 20px;margin-bottom:18px;display:flex;align-items:center;gap:16px;">'

        f'<div style="font-size:2rem;">👤</div>'

        f'<div style="flex:1;">'
        f'  <div style="font-size:1rem;font-weight:800;color:#ff9500;letter-spacing:1px;">'
        f'  DRAFT READY — YOUR REVIEW IS REQUIRED</div>'
        f'  <div style="font-size:0.78rem;color:#8aa3b8;margin-top:4px;">'
        f'  Brief: <code style="color:#ff9500;background:#1e1505;'
        f'  padding:1px 6px;border-radius:4px;word-break:break-all;">{brief_id}</code>'
        f'  &nbsp;·&nbsp; {channel}'
        f'  &nbsp;·&nbsp; {brand}'
        f'  &nbsp;·&nbsp; {persona}</div>'
        f'</div>'

        f'<div style="display:flex;gap:10px;align-items:center;">'
        f'  <div style="text-align:center;">'
        f'    <div style="font-size:1.5rem;font-weight:900;color:{score_color};">'
        f'    {judge_score:.2f}</div>'
        f'    <div style="font-size:0.65rem;color:#64748b;">JUDGE SCORE</div>'
        f'  </div>'
        f'  <div style="text-align:center;">'
        f'    <div style="font-size:1.1rem;font-weight:900;color:{comp_color};">'
        f'    {comp_label}</div>'
        f'    <div style="font-size:0.65rem;color:#64748b;">COMPLIANCE</div>'
        f'  </div>'
        f'</div>'

        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Main review layout ─────────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 1])

    with left_col:
        st.markdown(
            '<div style="font-size:0.72rem;color:#00d4ff;letter-spacing:2px;'
            'text-transform:uppercase;margin-bottom:8px;">✍️ Generated Draft</div>',
            unsafe_allow_html=True,
        )
        edited_draft = st.text_area(
            "Draft content (edit before approving)",
            value=draft,
            height=320,
            key="inline_draft_editor",
            label_visibility="collapsed",
        )

        # [image-based-campaign] Generated image + its compliance verdict.
        render_generated_image(draft_data.get("draft_metadata"))

        # Blog channel: offer the responsive HTML page (preview + download).
        if channel == "BLOG" and draft:
            _render_blog_page_section(api_base, brief_id)

    with right_col:
        # ── Compliance summary ─────────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.72rem;color:#00d4ff;letter-spacing:2px;'
            'text-transform:uppercase;margin-bottom:8px;">🛡️ Compliance</div>',
            unsafe_allow_html=True,
        )

        comp_bg = "#091a0f" if compliance else "#1a0909"
        comp_border = "#39ff14" if compliance else "#ff4444"
        st.markdown(
            f'<div style="background:{comp_bg};border:1px solid {comp_border};'
            f'border-radius:8px;padding:10px 12px;margin-bottom:10px;">'
            f'  <div style="font-size:1.4rem;font-weight:900;color:{comp_color};">{"✅ PASS" if compliance else "❌ FAIL"}</div>'
            f'  <div style="font-size:0.7rem;color:#64748b;margin-top:2px;">'
            f'  {len(violations)} violation(s) found</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if violations:
            with st.expander("View violations", expanded=True):
                for v in violations:
                    st.warning(v, icon="⚠️")

        # ── RAGAS scores ───────────────────────────────────────────────────────
        if ragas_scores:
            st.markdown(
                '<div style="font-size:0.72rem;color:#00d4ff;letter-spacing:2px;'
                'text-transform:uppercase;margin:12px 0 8px 0;">📊 RAGAS Scores</div>',
                unsafe_allow_html=True,
            )
            for metric, score in ragas_scores.items():
                label = metric.replace("_", " ").title()
                score_f = float(score)
                bar_color = "#39ff14" if score_f >= 0.7 else "#ff9500" if score_f >= 0.4 else "#ff4444"
                st.markdown(
                    f'<div style="margin-bottom:8px;">'
                    f'  <div style="display:flex;justify-content:space-between;'
                    f'  font-size:0.72rem;color:#8aa3b8;margin-bottom:3px;">'
                    f'  <span>{label}</span><span style="color:{bar_color};font-weight:700;">'
                    f'  {score_f:.2f}</span></div>'
                    f'  <div style="height:4px;background:#1e293b;border-radius:2px;">'
                    f'    <div style="width:{int(score_f*100)}%;height:100%;background:{bar_color};'
                    f'    border-radius:2px;"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Judge score visual ─────────────────────────────────────────────────
        st.markdown(
            f'<div style="background:#0b1525;border:1px solid #1b3558;border-radius:8px;'
            f'padding:10px 12px;margin-top:8px;">'
            f'  <div style="font-size:0.68rem;color:#64748b;margin-bottom:6px;">JUDGE SCORE</div>'
            f'  <div style="height:6px;background:#1e293b;border-radius:3px;overflow:hidden;">'
            f'    <div style="width:{int(judge_score*100)}%;height:100%;background:{score_color};'
            f'    border-radius:3px;"></div></div>'
            f'  <div style="font-size:1.3rem;font-weight:900;color:{score_color};margin-top:6px;">'
            f'  {judge_score:.2f}<span style="font-size:0.72rem;color:#64748b;"> / 1.00</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Decision form ──────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.72rem;color:#ff9500;letter-spacing:2px;'
        'text-transform:uppercase;border-bottom:1px solid #2d1f00;'
        'padding-bottom:8px;margin-bottom:14px;">🗳️ Submit Your Decision</div>',
        unsafe_allow_html=True,
    )

    with st.form("inline_review_form", clear_on_submit=False):
        reviewer = st.text_input(
            "Your Name / Email",
            value="reviewer@brand.com",
            key="inline_reviewer_id",
        )

        btn1, btn2, btn3 = st.columns(3)
        with btn1:
            approve = st.form_submit_button(
                "✅  Approve",
                use_container_width=True,
                type="primary",
            )
        with btn2:
            approve_edit = st.form_submit_button(
                "✏️  Approve with Edits",
                use_container_width=True,
            )
        with btn3:
            reject = st.form_submit_button(
                "❌  Reject",
                use_container_width=True,
            )

    # ── Handle decision ────────────────────────────────────────────────────────
    if approve:
        _submit_decision(api_base, brief_id, "approved", None, reviewer)
    elif approve_edit:
        if edited_draft.strip() == draft.strip():
            st.warning("No edits detected — use Approve instead.")
        else:
            _submit_decision(api_base, brief_id, "edited", edited_draft.strip(), reviewer)
    elif reject:
        _submit_decision(api_base, brief_id, "rejected", None, reviewer)


def _poll_pipeline_terminal(api_base: str, brief_id: str, max_seconds: int = 120) -> dict | None:
    """Wait until brief reaches indexed/rejected or timeout."""
    deadline = time.time() + max_seconds
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{api_base}/status/{brief_id}", timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") in ("indexed", "rejected"):
                return data
        except Exception:
            pass
        time.sleep(2)
    return None


def _render_blog_page_section(api_base: str, brief_id: str) -> None:
    """Preview + download the blog draft rendered as a responsive HTML page.
    Shown only for the 'blog' channel."""
    st.markdown(
        '<div style="font-size:0.72rem;color:#00d4ff;letter-spacing:2px;'
        'text-transform:uppercase;margin:14px 0 8px;">📰 Blog Page (HTML)</div>',
        unsafe_allow_html=True,
    )
    try:
        r = httpx.get(f"{api_base}/blog-page/{brief_id}", timeout=15.0)
        r.raise_for_status()
        page_html = r.text
    except Exception as exc:  # noqa: BLE001
        st.info(f"Blog HTML page not available yet: {exc}")
        return

    st.download_button(
        "⬇️ Download HTML page",
        data=page_html.encode("utf-8"),
        file_name=f"blog_{brief_id[:8]}.html",
        mime="text/html",
        use_container_width=True,
    )
    with st.expander("Preview rendered blog page", expanded=True):
        components.html(page_html, height=600, scrolling=True)


def _submit_decision(
    api_base: str, brief_id: str, decision: str,
    edits: str | None, reviewer: str,
) -> None:
    payload: dict = {"decision": decision, "reviewer": reviewer}
    if edits:
        payload["edits"] = edits

    _dec_colors = {"approved": "#39ff14", "edited": "#ff9500", "rejected": "#ff4444"}
    _dec_icons  = {"approved": "✅", "edited": "✏️", "rejected": "❌"}
    color = _dec_colors.get(decision, "#00d4ff")
    icon  = _dec_icons.get(decision, "📋")

    try:
        def _post_decision():
            r = httpx.post(
                f"{api_base}/decision/{brief_id}",
                json=payload,
                timeout=15.0,
            )
            r.raise_for_status()
            return r

        run_with_progress(
            f"Submitting decision: {decision}…",
            _post_decision,
            estimated_seconds=5,
        )
        st.session_state["gate_decision_made"] = True
        st.session_state["gate_brief_id"] = brief_id

        if decision in ("approved", "edited"):
            st.success(
                f"{icon} **Decision submitted: {decision.upper()}**\n\n"
                "Resuming pipeline — Curator is indexing your content…"
            )
            final = run_with_progress(
                "Waiting for Curator to finish…",
                lambda: _poll_pipeline_terminal(api_base, brief_id),
                estimated_seconds=30,
                sublabel="Step 7 · Index to knowledge base",
            )
            if final and final.get("status") == "indexed":
                doc_id = final.get("indexed_doc_id") or "—"
                st.success(
                    f"Pipeline complete — content indexed to the knowledge base.\n\n"
                    f"Document ID: `{doc_id}`"
                )
            elif final and final.get("status") == "rejected":
                st.warning("Brief ended with status **rejected**.")
            else:
                st.info(
                    "Pipeline is still running in the background. "
                    "Check **Search Brief** or **Audit** for the final status."
                )
        else:
            st.error(
                f"{icon} **Draft rejected.** The pipeline has ended for this brief.\n\n"
                "You can submit a new brief with an updated key message below."
            )

        st.markdown(
            f'<div style="background:#0b1525;border:1px solid {color};border-left:4px solid {color};'
            f'border-radius:10px;padding:12px 16px;margin-top:12px;">'
            f'<div style="color:{color};font-weight:700;">Brief ID: <code style="color:{color};">'
            f'{brief_id}</code></div>'
            f'<div style="color:#8aa3b8;font-size:0.78rem;margin-top:4px;">'
            f'Decision: {decision.upper()} · Reviewer: {reviewer}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button("Submit Another Brief", type="secondary"):
            for k in ("gate_brief_id", "gate_decision_made", "active_brief_id"):
                st.session_state.pop(k, None)
            st.rerun()

    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
    except Exception as exc:
        st.error(f"Decision submission failed: {exc}")


# ── Main render ────────────────────────────────────────────────────────────────
def render() -> None:
    inject_progress_css()
    api_base = st.session_state.get("api_base", "http://localhost:8000/api/v1")

    # ── If a pipeline is paused at Human Gate, show review panel ──────────────
    gate_brief_id    = st.session_state.get("gate_brief_id")
    gate_decision_ok = st.session_state.get("gate_decision_made", False)

    if gate_brief_id and not gate_decision_ok:
        # Show the tracker state hint
        st.markdown(
            '<div style="font-size:0.72rem;color:#475569;letter-spacing:2px;'
            'text-transform:uppercase;margin-bottom:10px;">📡 Pipeline Paused — Human Gate Active</div>',
            unsafe_allow_html=True,
        )
        _render_review_panel(api_base, gate_brief_id)
        st.divider()
        st.markdown(
            '<div style="color:#64748b;font-size:0.82rem;margin-bottom:8px;">'
            'Submit a new brief while you wait, or clear the pending review above.</div>',
            unsafe_allow_html=True,
        )

    elif gate_decision_ok and gate_brief_id:
        try:
            resp = httpx.get(f"{api_base}/status/{gate_brief_id}", timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "unknown")
                if status == "indexed":
                    st.success(
                        f"Brief `{gate_brief_id}` is **indexed**. "
                        f"Doc ID: `{data.get('indexed_doc_id') or '—'}`"
                    )
                elif status == "curating":
                    st.info("Curator is still indexing — refresh in a few seconds.")
                elif status == "rejected":
                    st.warning("This brief was **rejected**.")
                else:
                    st.info(f"Brief status: **{status}**")
        except Exception:
            st.info("Decision recorded — check **Search Brief** or **Audit** for status.")
        if st.button("✨ Submit a New Brief", type="primary"):
            for k in ("gate_brief_id", "gate_decision_made", "active_brief_id"):
                st.session_state.pop(k, None)
            st.rerun()
        return

    elif gate_decision_ok:
        # Decision was just made — clear state on next interaction
        if st.button("✨ Submit a New Brief", type="primary"):
            for k in ("gate_brief_id", "gate_decision_made", "active_brief_id"):
                st.session_state.pop(k, None)
            st.rerun()
        return

    # ── Brief submission form ──────────────────────────────────────────────────
    st.caption(
        "Submit a campaign brief — the AI pipeline runs 7 stages automatically. "
        "Typical time: **30–60 s** with a 3B model, 60–90 s with 8B."
    )

    api_base = st.session_state.get("api_base", "http://localhost:8000/api/v1")
    persona_names, persona_displays = _fetch_persona_options(api_base)

    with st.form("brief_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            brand = st.text_input("Brand Name *", placeholder="e.g. GlowBrand")
        with col2:
            channel = st.selectbox(
                "Channel *",
                options=["email", "linkedin", "social", "ad", "blog"],
                format_func=lambda x: {
                    "email":    "📧 Email",
                    "linkedin": "💼 LinkedIn",
                    "social":   "📱 Social Media",
                    "ad":       "📣 Digital Ad",
                    "blog":     "📝 Blog Post",
                }[x],
            )
        persona = st.selectbox(
            "Target Persona *",
            options=persona_names,
            format_func=lambda n: persona_displays.get(n, n),
            help="Audience segment with age range and spend profile — pick the best match for your campaign.",
        )
        key_message = st.text_area(
            "Key Message *",
            placeholder="What is the core message? (min 10 characters)",
            height=100,
        )
        constraints = st.text_area(
            "Extra Constraints (JSON, optional)",
            value="{}",
            height=52,
            help='e.g. {"required_phrases": ["shop now"]}',
        )
        # [image-based-campaign] Optional reference image → campaign (image → text).
        uploaded_image = st.file_uploader(
            "Reference Image (optional) — the AI will read it and ground the campaign in it",
            type=["png", "jpg", "jpeg", "webp"],
        )
        # [image-based-campaign] Per-brief switch for image OUTPUT.
        # NOTE: this MUST be keyed. An *unkeyed* checkbox the user never toggles
        # does not commit its `value=True` default to the form on the very FIRST
        # submit of a freshly-rendered form (Streamlit 1.58), so the first brief
        # was silently sent with generate_image=False and no image was produced;
        # every later submit worked. A `key` persists the value in session_state
        # from the first render, so the first submit reads True correctly.
        st.session_state.setdefault("gen_image_flag", True)
        generate_image = st.checkbox(
            "🖼️  Generate a campaign image",
            key="gen_image_flag",
            help="Unchecked returns copy only — no image is generated and no image "
                 "provider is called. A reference image you upload is still read either way.",
        )
        submitted = st.form_submit_button(
            "🚀 Submit Brief & Run Pipeline",
            use_container_width=True,
            type="primary",
        )

    if not submitted:
        return

    # ── Validation ────────────────────────────────────────────────────────────
    if not brand.strip():
        st.error("Brand name is required.")
        return
    # [image-based-campaign] With a reference image, key message may be derived by vision.
    has_image = uploaded_image is not None
    if not has_image and (not key_message.strip() or len(key_message.strip()) < 10):
        st.error("Key message must be at least 10 characters.")
        return
    try:
        constraints_dict = json.loads(constraints) if constraints.strip() else {}
    except json.JSONDecodeError:
        st.error("Extra Constraints must be valid JSON, e.g. {}")
        return

    payload = {
        "brand":       brand.strip(),
        "channel":     channel,
        "persona":     persona,
        "key_message": key_message.strip(),
        "constraints": constraints_dict,
        "generate_image": generate_image,
    }

    # [image-based-campaign] Multipart submit when a reference image was uploaded,
    # otherwise the original JSON brief endpoint.
    def _submit_brief():
        if has_image:
            files = {"image": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
            form = {
                "brand": payload["brand"], "channel": payload["channel"],
                "persona": payload["persona"], "key_message": payload["key_message"],
                "constraints": json.dumps(constraints_dict),
                "generate_image": str(generate_image).lower(),
            }
            resp = httpx.post(f"{api_base}/brief/image", data=form, files=files, timeout=30.0)
        else:
            resp = httpx.post(f"{api_base}/brief", json=payload, timeout=20.0)
        resp.raise_for_status()
        return resp.json()

    # ── Submit brief to API ───────────────────────────────────────────────────
    try:
        data = run_with_progress(
            "Submitting brief…",
            _submit_brief,
            estimated_seconds=6,
            sublabel="Starting AI pipeline",
        )
    except httpx.ConnectError:
        st.error(
            f"Cannot connect to API at `{api_base}`. "
            "Start the backend first: `python main.py`"
        )
        return
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
        return

    brief_id = data["brief_id"]
    st.session_state["active_brief_id"] = brief_id
    st.session_state["review_brief_id"] = brief_id
    st.success(f"Brief accepted — ID: `{brief_id}`")

    # ── 7-step live tracker ───────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.72rem;color:#475569;letter-spacing:2px;'
        'text-transform:uppercase;margin:14px 0 8px 2px;">📡 Live Pipeline Progress</div>',
        unsafe_allow_html=True,
    )

    step_phs    = [st.empty() for _ in range(7)]
    progress_ph = st.empty()
    result_ph   = st.empty()

    steps      = _fresh_steps()
    start_time = time.time()

    steps[0].status  = "done"
    steps[0].elapsed = 0.0
    _render_all(step_phs, steps, 0.0, progress_ph)

    # ── SSE stream ────────────────────────────────────────────────────────────
    trace_lines:     list[str] = []
    gate_fired:      bool      = False

    try:
        timeout = httpx.Timeout(connect=10.0, read=_PIPELINE_TIMEOUT_S,
                                write=10.0, pool=5.0)
        with httpx.Client(timeout=timeout) as client:
            with client.stream("GET", f"{api_base}/stream/{brief_id}") as resp_sse:
                resp_sse.raise_for_status()

                for line in resp_sse.iter_lines():
                    now     = time.time()
                    elapsed = now - start_time

                    if not line:
                        continue

                    if line.startswith("data:"):
                        raw = line[5:].strip()
                        try:
                            evt = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        msg = evt.get("message", "")
                        if msg and not msg.startswith("__"):
                            trace_lines.append(msg)
                            _update_steps(steps, msg, now, start_time)
                            _render_all(step_phs, steps, elapsed, progress_ph)

                    elif line.startswith("event:"):
                        etype = line[6:].strip()

                        if etype == "human_action_required":
                            if steps[4].status == "running":
                                steps[4].status  = "done"
                                steps[4].elapsed = elapsed
                            steps[5].status = "waiting"
                            steps[5].detail = "Your approval is needed — see review panel below"
                            _render_all(step_phs, steps, elapsed, progress_ph)
                            gate_fired = True
                            break

                        elif etype == "pipeline_complete":
                            for s in steps:
                                if s.status in ("pending", "running", "waiting"):
                                    s.status  = "done"
                                    s.elapsed = elapsed
                            _render_all(step_phs, steps, elapsed, progress_ph)
                            result_ph.success(
                                f"Pipeline complete in **{elapsed:.0f}s**! "
                                "Check **🔍 Search Brief** or the Audit log."
                            )
                            break

                        elif etype == "done":
                            for s in steps:
                                if s.status in ("pending", "running"):
                                    s.status  = "done"
                                    s.elapsed = elapsed
                            _render_all(step_phs, steps, elapsed, progress_ph)
                            break

                        elif etype == "error":
                            for s in steps:
                                if s.status == "running":
                                    s.status  = "error"
                                    s.elapsed = elapsed
                            _render_all(step_phs, steps, elapsed, progress_ph)
                            result_ph.error(
                                "Pipeline error — check backend terminal logs."
                            )
                            break

    except httpx.ReadTimeout:
        elapsed = time.time() - start_time
        for s in steps:
            if s.status == "running":
                s.detail = "Stream timed out — still running in backend"
        _render_all(step_phs, steps, elapsed, progress_ph)
        result_ph.warning(
            f"Stream timed out after {elapsed:.0f}s. Brief is still processing.\n\n"
            f"Brief ID: `{brief_id}`  — check **🔍 Search Brief** in a moment.",
            icon="⚠️",
        )
    except httpx.ConnectError:
        result_ph.error("Lost connection to API stream. Is the backend still running?")
    except Exception as exc:
        result_ph.error(f"Stream error: {exc}")

    # ── Gate fired: store brief_id and rerun so the top-of-render block
    # shows the review panel cleanly (avoids duplicate widget-key error
    # that occurs when rendering it both here AND at the top of render()).
    if gate_fired:
        st.session_state["gate_brief_id"] = brief_id
        st.session_state["review_brief_id"] = brief_id
        st.session_state["active_brief_id"] = brief_id
        from ui.navigation import MAIN_TAB_STUDIO, STUDIO_TAB_CREATE, navigate_to

        navigate_to(MAIN_TAB_STUDIO, STUDIO_TAB_CREATE)
        st.rerun()
    else:
        show_ring(progress_ph, 100, "Pipeline finished", "All steps complete", height=240)
        time.sleep(0.8)

    # ── Full trace collapsible ────────────────────────────────────────────────
    if trace_lines:
        with st.expander("📋 Full agent trace log", expanded=False):
            for ln in trace_lines:
                st.text(ln)
