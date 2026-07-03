from __future__ import annotations

import httpx
import streamlit as st


def _compliance_badge(pass_: bool) -> str:
    return "🟢 PASS" if pass_ else "🔴 FAIL"


def render() -> None:
    st.markdown('<div style="color:#64748b;font-size:0.88rem;margin-bottom:16px;">Inspect the generated draft, compliance results, and submit your approval decision.</div>', unsafe_allow_html=True)

    api_base = st.session_state.get("api_base", "http://localhost:8000/api/v1")

    brief_id = st.text_input(
        "Brief ID",
        value=st.session_state.get("active_brief_id") or "",
        placeholder="Paste brief UUID here",
    )

    if not brief_id:
        st.info("Enter a Brief ID above or submit a brief from the **Create Content** page.")
        return

    with st.spinner("Fetching status…"):
        try:
            resp = httpx.get(f"{api_base}/status/{brief_id}", timeout=10.0)
            if resp.status_code == 404:
                st.error(f"Brief `{brief_id}` not found.")
                return
            resp.raise_for_status()
            data = resp.json()
        except httpx.ConnectError:
            st.error(f"Cannot connect to API at `{api_base}`.")
            return
        except httpx.HTTPStatusError as exc:
            st.error(f"API error {exc.response.status_code}: {exc.response.text}")
            return

    st.session_state["active_brief_id"] = brief_id

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Channel", data.get("channel", "—").upper())
    col2.metric("Brand", data.get("brand", "—"))
    col3.metric("Persona", data.get("persona", "—"))
    col4.metric("Status", data.get("status", "—").capitalize())

    st.divider()

    draft = data.get("draft", "")
    compliance_pass = data.get("compliance_pass", False)
    judge_score = data.get("judge_score", 0.0)
    rule_violations = data.get("rule_violations", [])
    ragas_scores = data.get("ragas_scores", {})
    human_decision = data.get("human_decision")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("### Draft Content")
        if draft:
            edited_draft = st.text_area(
                "Edit draft before approving (optional):",
                value=draft,
                height=350,
                key="draft_editor",
            )
        else:
            st.info("No draft available yet. The pipeline may still be running.")
            edited_draft = ""

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

    if human_decision:
        st.info(f"Decision already submitted: **{human_decision.upper()}**")
        return

    if not draft:
        return

    st.divider()
    st.markdown("### Your Decision")

    reviewer_id = st.text_input("Your Name / Email", value="reviewer@brand.com")

    col_approve, col_edit, col_reject = st.columns(3)

    with col_approve:
        if st.button("✅ Approve", use_container_width=True, type="primary"):
            _submit_decision(
                api_base=api_base,
                brief_id=brief_id,
                decision="approved",
                edits=None,
                reviewer=reviewer_id,
            )

    with col_edit:
        if st.button("✏️ Approve with Edits", use_container_width=True):
            if edited_draft.strip() == draft.strip():
                st.warning("No edits detected. Use **Approve** instead.")
            else:
                _submit_decision(
                    api_base=api_base,
                    brief_id=brief_id,
                    decision="edited",
                    edits=edited_draft.strip(),
                    reviewer=reviewer_id,
                )

    with col_reject:
        if st.button("❌ Reject", use_container_width=True):
            _submit_decision(
                api_base=api_base,
                brief_id=brief_id,
                decision="rejected",
                edits=None,
                reviewer=reviewer_id,
            )


def _submit_decision(
    api_base: str,
    brief_id: str,
    decision: str,
    edits: str | None,
    reviewer: str,
) -> None:
    payload = {"decision": decision, "reviewer": reviewer}
    if edits:
        payload["edits"] = edits

    try:
        resp = httpx.post(
            f"{api_base}/decision/{brief_id}",
            json=payload,
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        st.success(data.get("message", f"Decision '{decision}' submitted."))
    except httpx.ConnectError:
        st.error("Cannot connect to API.")
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
