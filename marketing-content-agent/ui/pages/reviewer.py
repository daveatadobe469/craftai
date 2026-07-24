from __future__ import annotations

import base64

import httpx
import streamlit as st
import streamlit.components.v1 as components

from ui.components.image_review import render_generated_image  # [image-based-campaign]
from ui.components.progress import inject_progress_css, run_with_progress


def _compliance_badge(pass_: bool) -> str:
    return "🟢 PASS" if pass_ else "🔴 FAIL"


def _render_blog_page_section(api_base: str, brief_id: str) -> None:
    """Preview + download the blog draft rendered as a responsive HTML page."""
    st.markdown("### 📰 Blog Page Preview")
    try:
        r = httpx.get(f"{api_base}/blog-page/{brief_id}", timeout=15.0)
        r.raise_for_status()
        page_html = r.text
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not render blog page: {exc}")
        return

    st.download_button(
        "⬇️ Download HTML",
        data=page_html.encode("utf-8"),
        file_name=f"blog_{brief_id[:8]}.html",
        mime="text/html",
        use_container_width=False,
    )
    with st.expander("Preview rendered page", expanded=True):
        components.html(page_html, height=700, scrolling=True)


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

    # [mcp-email] Surface the outcome of an email action from the previous rerun.
    _render_email_result(brief_id)

    status = str(data.get("status", "")).lower()
    can_review = (not human_decision) and bool(draft) and status == "awaiting_review"

    if draft and str(data.get("channel", "")).lower() == "blog":
        _render_blog_page_section(api_base, brief_id)
        st.divider()

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
            # [mcp-email] Delivery option — honoured on Approve / Approve with Edits.
            # Email channel → .eml / SMTP; every other channel → downloadable HTML.
            _is_email = str(data.get("channel", "")).lower() == "email"
            em1, em2 = st.columns([1, 1])
            if _is_email:
                delivery_label = em1.radio(
                    "📧 On approval",
                    ["Don't email", "Download .eml", "Send email"],
                    index=1,  # default to Download .eml for email briefs
                    help="Download .eml opens in Mail/Outlook (no SMTP). "
                         "Send email delivers via SMTP (needs SMTP_HOST configured).",
                )
                delivery_to = em2.text_input("Email to", value="reviewer@brand.com")
            else:
                delivery_label = em1.radio(
                    "📄 On approval",
                    ["Don't download", "Download HTML"],
                    index=1,  # default to HTML download for non-email briefs
                    help="Download the campaign as a standalone HTML file.",
                )
                delivery_to = ""
            b1, b2, b3 = st.columns(3)
            approve = b1.form_submit_button("✅ Approve", type="primary", use_container_width=True)
            approve_edit = b2.form_submit_button("✏️ Approve with Edits", use_container_width=True)
            reject = b3.form_submit_button("❌ Reject", use_container_width=True)

        email = _delivery_choice(delivery_label, delivery_to)  # [mcp-email]

        if approve:
            _submit_decision(api_base, brief_id, "approved", None, reviewer, email=email)
        elif approve_edit:
            if edited_draft.strip() == draft.strip():
                st.warning("No edits detected — use Approve instead.")
            else:
                _submit_decision(api_base, brief_id, "edited", edited_draft.strip(), reviewer, email=email)
        elif reject:
            _submit_decision(api_base, brief_id, "rejected", None, reviewer)
    elif draft:
        # [mcp-email] Brief is already decided (or not in review) — the approval form is
        # gone, but you can still download/send the finished campaign from this panel.
        _render_email_panel(api_base, brief_id, str(data.get("channel", "")).lower() == "email")
        if not human_decision and status != "awaiting_review":
            st.info(
                f"This brief is in **{status or 'unknown'}** state — not awaiting review, "
                "so no decision can be submitted here."
            )


def _submit_decision(
    api_base: str, brief_id: str, decision: str, edits: str | None, reviewer: str,
    email: dict | None = None,  # [mcp-email]
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
        # [mcp-email] Fire the email AFTER the decision is recorded, so any edits are
        # already persisted and get reflected in the campaign email.
        if email and decision in ("approved", "edited"):
            _trigger_email(api_base, brief_id, email["mode"], email["to"])
        # Clear cached decision state and refresh the view.
        st.session_state.pop("review_brief_id", None)
        st.session_state["review_brief_id"] = brief_id
        st.rerun()
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Decision submission failed: {exc}")


# ── [mcp-email] Campaign delivery helpers (remove this block with the feature) ───
def _render_email_panel(api_base: str, brief_id: str, is_email: bool) -> None:
    """Standalone delivery control for briefs no longer in review (approved / complete)."""
    st.divider()
    if is_email:
        st.markdown("### 📧 Email this campaign")
        st.caption("This brief is past review — you can still download or send its campaign email.")
        with st.form(f"email_panel_{brief_id}", clear_on_submit=False):
            c1, c2 = st.columns([1, 1])
            mode_label = c1.radio(
                "Delivery",
                ["Download .eml", "Send email"],
                help="Download .eml opens in Mail/Outlook (no SMTP). Send needs SMTP_HOST configured.",
            )
            to = c2.text_input("Email to", value="reviewer@brand.com")
            go = st.form_submit_button("📧 Generate / Send", type="primary")
        if go:
            _trigger_email(api_base, brief_id, "send" if mode_label == "Send email" else "eml", to)
            st.rerun()
    else:
        st.markdown("### 📄 Download this campaign")
        st.caption("This brief is past review — download the campaign as a standalone HTML file.")
        with st.form(f"html_panel_{brief_id}", clear_on_submit=False):
            go = st.form_submit_button("📄 Generate HTML", type="primary")
        if go:
            _trigger_email(api_base, brief_id, "html", "")
            st.rerun()


def _delivery_choice(label: str, to: str) -> dict | None:
    """Map a radio label to a delivery request, or None for the 'do nothing' options."""
    mode = {"Download .eml": "eml", "Send email": "send", "Download HTML": "html"}.get(label)
    return {"mode": mode, "to": (to or "").strip()} if mode else None


def _trigger_email(api_base: str, brief_id: str, mode: str, to: str) -> None:
    """Call the delivery endpoint (the MCP client) and stash the result for the next rerun."""
    try:
        r = httpx.post(
            f"{api_base}/email/{brief_id}",
            json={"mode": mode, "to": to or None},
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001 — delivery is best-effort, never blocks the decision
        st.session_state["email_flash"] = ("error", f"Delivery step failed: {exc}")
        return

    if mode == "send":
        sent = data.get("status") == "sent"
        st.session_state["email_flash"] = (
            "success" if sent else "error",
            data.get("detail", "Send attempted."),
        )
    elif mode == "html":
        st.session_state["campaign_download"] = {
            "brief_id": brief_id,
            "data": (data.get("html", "") or "").encode("utf-8"),
            "filename": f"campaign_{brief_id[:8]}.html",
            "mime": "text/html",
            "label": "⬇️ Download campaign HTML",
        }
        st.session_state["email_flash"] = ("success", "Campaign HTML ready — download below.")
    else:  # eml
        st.session_state["campaign_download"] = {
            "brief_id": brief_id,
            "data": base64.b64decode(data.get("eml_base64", "") or ""),
            "filename": f"campaign_{brief_id[:8]}.eml",
            "mime": "message/rfc822",
            "label": "⬇️ Download campaign .eml",
        }
        st.session_state["email_flash"] = ("success", "Campaign .eml ready — download below.")


def _render_email_result(brief_id: str) -> None:
    """Show the flash message + download button (.eml or .html) from the last action."""
    flash = st.session_state.pop("email_flash", None)
    if flash:
        (st.success if flash[0] == "success" else st.error)(flash[1])
    dl = st.session_state.get("campaign_download")
    if dl and dl.get("brief_id") == brief_id and dl.get("data"):
        st.download_button(
            dl["label"],
            data=dl["data"],
            file_name=dl["filename"],
            mime=dl["mime"],
            key=f"dl_{brief_id}",
        )
