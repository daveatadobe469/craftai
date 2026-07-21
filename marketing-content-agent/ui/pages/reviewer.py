from __future__ import annotations

import httpx
import streamlit as st

from ui.components.image_review import render_generated_image  # [image-based-campaign]
from ui.components.progress import inject_progress_css, run_with_progress


def _compliance_badge(pass_: bool) -> str:
    return "🟢 PASS" if pass_ else "🔴 FAIL"


def render() -> None:
    inject_progress_css()
    st.markdown(
        '<div style="color:#64748b;font-size:0.88rem;margin-bottom:16px;">'
        "Search a brief by ID to view its pipeline status, approval decision, and final content."
        "</div>",
        unsafe_allow_html=True,
    )

    api_base = st.session_state.get("api_base", "http://localhost:8000/api/v1")

    # Auto-load the latest brief when arriving from Create Brief
    if not st.session_state.get("review_brief_id") and st.session_state.get("active_brief_id"):
        st.session_state["review_brief_id"] = st.session_state["active_brief_id"]

    default_brief = (
        st.session_state.get("review_brief_id")
        or st.session_state.get("active_brief_id")
        or ""
    )

    with st.form("review_brief_lookup", clear_on_submit=False):
        brief_input = st.text_input(
            "Brief ID",
            value=str(default_brief),
            placeholder="Paste brief UUID here",
        )
        load = st.form_submit_button("🔍 Load Brief", type="primary", use_container_width=True)

    if load:
        brief_id = brief_input.strip()
        if not brief_id:
            st.warning("Enter a Brief ID to load.")
            return
        st.session_state["review_brief_id"] = brief_id
        st.session_state["active_brief_id"] = brief_id
        st.rerun()

    brief_id = str(st.session_state.get("review_brief_id") or "").strip()
    if not brief_id:
        st.info("Enter a Brief ID above and click **Load Brief**, or submit a brief from **Create Brief**.")
        return

    try:
        def _fetch_status():
            resp = httpx.get(f"{api_base}/status/{brief_id}", timeout=10.0)
            if resp.status_code == 404:
                raise LookupError(f"Brief `{brief_id}` not found.")
            resp.raise_for_status()
            return resp.json()

        data = run_with_progress(
            "Fetching brief status…",
            _fetch_status,
            estimated_seconds=4,
        )
    except LookupError as exc:
        st.error(str(exc))
        return
    except httpx.ConnectError:
        st.error(f"Cannot connect to API at `{api_base}`.")
        return
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
        return

    st.session_state["active_brief_id"] = brief_id

    human_decision = data.get("human_decision")
    draft = data.get("draft", "")
    compliance_pass = data.get("compliance_pass", False)
    judge_score = data.get("judge_score", 0.0)
    rule_violations = data.get("rule_violations", [])
    ragas_scores = data.get("ragas_scores", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Channel", data.get("channel", "—").upper())
    col2.metric("Brand", data.get("brand", "—"))
    col3.metric("Persona", data.get("persona", "—"))
    col4.metric("Status", data.get("status", "—").capitalize())
    if human_decision:
        col5.metric("Decision", human_decision.upper())
    else:
        col5.metric("Decision", "Pending")

    st.divider()

    status = str(data.get("status", "")).lower()
    can_review = (not human_decision) and bool(draft) and status == "awaiting_review"

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("### Brief Content")
        if human_decision:
            st.caption(f"Decision already taken: **{human_decision.upper()}**")
        elif can_review:
            st.caption("Awaiting your review — edit the draft if needed, then choose an action below.")
        elif draft:
            st.caption("Draft generated.")
        if draft:
            edited_draft = st.text_area(
                "Brief content",
                value=draft,
                height=350,
                key=f"brief_view_{brief_id}",
                disabled=not can_review,
                label_visibility="collapsed",
            )
        else:
            edited_draft = ""
            st.info("No brief content available yet. The pipeline may still be running.")

        # [image-based-campaign] Generated image + its compliance verdict.
        render_generated_image(data.get("draft_metadata"))

    with col_b:
        st.markdown("### Compliance Summary")
        st.markdown(f"**Rule Check:** {_compliance_badge(compliance_pass)}")
        st.metric("Judge Score", f"{judge_score:.2f}", help="LLM-as-judge holistic score (0–1)")

        if rule_violations:
            st.markdown("**Violations:**")
            for v in rule_violations:
                st.warning(v)
        else:
            st.success("No rule violations.")

        if ragas_scores:
            st.markdown("**RAGAS Scores:**")
            for metric, score in ragas_scores.items():
                st.progress(float(score), text=f"{metric.replace('_', ' ').title()}: {score:.2f}")

    # ── Human review decision (recovery path if you left the Create Brief tab) ──
    if can_review:
        st.divider()
        st.markdown("### 🗳️ Submit Your Decision")
        with st.form(f"review_decision_{brief_id}", clear_on_submit=False):
            reviewer = st.text_input("Your Name / Email", value="reviewer@brand.com")
            b1, b2, b3 = st.columns(3)
            approve = b1.form_submit_button("✅ Approve", type="primary", use_container_width=True)
            approve_edit = b2.form_submit_button("✏️ Approve with Edits", use_container_width=True)
            reject = b3.form_submit_button("❌ Reject", use_container_width=True)

        if approve:
            _submit_decision(api_base, brief_id, "approved", None, reviewer)
        elif approve_edit:
            if edited_draft.strip() == draft.strip():
                st.warning("No edits detected — use Approve instead.")
            else:
                _submit_decision(api_base, brief_id, "edited", edited_draft.strip(), reviewer)
        elif reject:
            _submit_decision(api_base, brief_id, "rejected", None, reviewer)
    elif not human_decision and draft and status != "awaiting_review":
        st.divider()
        st.info(
            f"This brief is in **{status or 'unknown'}** state — not awaiting review, "
            "so no decision can be submitted here."
        )


def _submit_decision(
    api_base: str, brief_id: str, decision: str, edits: str | None, reviewer: str
) -> None:
    payload: dict = {"decision": decision, "reviewer": reviewer}
    if edits:
        payload["edits"] = edits
    try:
        r = httpx.post(f"{api_base}/decision/{brief_id}", json=payload, timeout=15.0)
        r.raise_for_status()
        if decision == "rejected":
            st.error(f"Brief rejected. Reviewer: {reviewer}")
        else:
            st.success(
                f"Decision **{decision}** submitted — the pipeline is resuming and the "
                "Curator will index the content. Reload the brief to see status **complete**."
            )
        # Clear cached decision state and refresh the view.
        st.session_state.pop("review_brief_id", None)
        st.session_state["review_brief_id"] = brief_id
        st.rerun()
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Decision submission failed: {exc}")
