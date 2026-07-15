from __future__ import annotations

import httpx
import pandas as pd
import streamlit as st

from ui.components.progress import inject_progress_css, run_with_progress, show_ring


def _id_column_config() -> dict:
    """Wide monospace columns so UUIDs display and export in full."""
    return {
        "Brief ID": st.column_config.TextColumn("Brief ID", width="large"),
        "Draft ID": st.column_config.TextColumn("Draft ID", width="large"),
    }


def _download_csv(df: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


def _get(api_base: str, path: str, params: dict | None = None) -> dict | list | None:
    try:
        r = httpx.get(f"{api_base}{path}", params=params or {}, timeout=8.0)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        return None
    except Exception:
        return None


def render() -> None:
    inject_progress_css()
    st.markdown('<div style="color:#64748b;font-size:0.88rem;margin-bottom:16px;">Full traceability of every brief, draft, compliance check, and human decision.</div>', unsafe_allow_html=True)

    api_base = st.session_state.get("api_base", "http://localhost:8000/api/v1")

    stats = run_with_progress(
        "Loading audit dashboard…",
        lambda: _get(api_base, "/audit/stats"),
        estimated_seconds=3,
        sublabel="Fetching pipeline statistics",
    )
    if stats is None:
        st.error("Cannot reach API. Make sure the backend is running.")
        return

    # ── Pipeline KPI metrics ──────────────────────────────────────────────────
    st.markdown("### 📊 Pipeline Overview")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Briefs",      stats.get("total_briefs", 0))
    k2.metric("Total Drafts",      stats.get("total_drafts", 0))
    k3.metric("Approved",          stats.get("approved_count", 0))
    k4.metric("Rejected",          stats.get("rejected_count", 0))
    k5.metric("Avg Judge Score",   f"{stats.get('avg_judge_score', 0):.2f}")
    k6.metric("Avg Revisions",     f"{stats.get('avg_revisions', 0):.1f}")

    # ── Event type breakdown ──────────────────────────────────────────────────
    events_by_type: dict = stats.get("events_by_type", {})
    if events_by_type:
        with st.expander("Event breakdown by type", expanded=False):
            df_evt = pd.DataFrame(
                [{"Event Type": k, "Count": v} for k, v in sorted(events_by_type.items())]
            )
            st.dataframe(df_evt, use_container_width=True, hide_index=True)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_briefs, tab_drafts, tab_events = st.tabs([
        "📝 Briefs",
        "📄 Drafts",
        "🔍 Audit Events",
    ])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1 — Briefs
    # ────────────────────────────────────────────────────────────────────────
    with tab_briefs:
        st.markdown("#### All Submitted Briefs")

        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            status_filter = st.selectbox(
                "Status", ["", "pending", "processing", "complete", "rejected"],
                format_func=lambda x: "— All —" if not x else x.capitalize(),
                key="brief_status_filter",
            )
        with filter_col2:
            brand_filter = st.text_input("Brand", placeholder="Filter by brand…", key="brief_brand_filter")
        with filter_col3:
            channel_filter = st.selectbox(
                "Channel", ["", "email", "linkedin", "social", "ad", "blog"],
                format_func=lambda x: "— All —" if not x else x.capitalize(),
                key="brief_channel_filter",
            )

        params = {}
        if status_filter:
            params["status"] = status_filter
        if brand_filter.strip():
            params["brand"] = brand_filter.strip()
        if channel_filter:
            params["channel"] = channel_filter

        briefs = run_with_progress(
            "Loading briefs…",
            lambda: _get(api_base, "/audit/briefs", params) or [],
            estimated_seconds=4,
        )
        if not briefs:
            st.info("No briefs found. Submit a campaign brief from the **Create Content** tab.")
        else:
            df = pd.DataFrame(briefs)
            df = df.rename(columns={
                "brief_id": "Brief ID", "brand": "Brand", "channel": "Channel",
                "persona": "Persona", "key_message": "Key Message",
                "status": "Status", "created_at": "Created",
            })

            _status_colour = {
                "pending": "🟡", "processing": "🔵",
                "complete": "🟢", "rejected": "🔴",
                "awaiting_review": "🟠",
            }
            df["Status"] = df["Status"].apply(lambda s: f"{_status_colour.get(s, '⚪')} {s}")

            display_cols = ["Brief ID", "Brand", "Channel", "Persona", "Status", "Created", "Key Message"]
            _download_csv(df[display_cols], "craftai_briefs.csv", "⬇️ Download briefs (CSV)")
            st.dataframe(
                df[display_cols],
                column_config=_id_column_config(),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"{len(briefs)} brief(s) shown")

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2 — Drafts
    # ────────────────────────────────────────────────────────────────────────
    with tab_drafts:
        st.markdown("#### All Draft Records")

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            decision_filter = st.selectbox(
                "Human Decision",
                ["", "approved", "edited", "rejected"],
                format_func=lambda x: "— All —" if not x else x.capitalize(),
                key="draft_decision_filter",
            )
        with d_col2:
            brief_id_filter = st.text_input(
                "Brief ID (full or partial)", placeholder="Filter by brief ID…",
                key="draft_brief_filter"
            )

        d_params = {}
        if decision_filter:
            d_params["human_decision"] = decision_filter
        if brief_id_filter.strip():
            d_params["brief_id"] = brief_id_filter.strip()

        drafts = run_with_progress(
            "Loading drafts…",
            lambda: _get(api_base, "/audit/drafts", d_params) or [],
            estimated_seconds=4,
        )
        if not drafts:
            st.info("No drafts found yet.")
        else:
            df_d = pd.DataFrame(drafts)
            df_d["compliance_pass"] = df_d["compliance_pass"].apply(lambda v: "✅" if v else "❌")
            df_d["judge_score"] = df_d["judge_score"].apply(
                lambda v: f"{v:.2f}" if v is not None else "—"
            )

            _dec_colour = {"approved": "🟢", "edited": "🟡", "rejected": "🔴", None: "⏳"}
            df_d["human_decision"] = df_d["human_decision"].apply(
                lambda v: f"{_dec_colour.get(v, '⚪')} {v or 'pending'}"
            )

            df_d = df_d.rename(columns={
                "draft_id": "Draft ID", "brief_id": "Brief ID",
                "revision_count": "Revisions", "judge_score": "Judge Score",
                "compliance_pass": "Compliant", "human_decision": "Decision",
                "reviewed_by": "Reviewer", "created_at": "Created",
            })
            display_cols = [
                "Draft ID", "Brief ID", "Revisions", "Judge Score",
                "Compliant", "Decision", "Reviewer", "Created",
            ]
            _download_csv(df_d[display_cols], "craftai_drafts.csv", "⬇️ Download drafts (CSV)")
            st.dataframe(
                df_d[display_cols],
                column_config=_id_column_config(),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"{len(drafts)} draft(s) shown")

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3 — Audit Events
    # ────────────────────────────────────────────────────────────────────────
    with tab_events:
        st.markdown("#### Raw Audit Event Log")

        meta = _get(api_base, "/audit/event-types")
        event_types: list[str] = (meta or {}).get("event_types", [])
        actors: list[str] = (meta or {}).get("actors", [])

        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        with e_col1:
            evt_brief_id = st.text_input(
                "Brief ID", placeholder="Filter by brief ID…", key="evt_brief_filter"
            )
        with e_col2:
            evt_type = st.selectbox(
                "Event Type",
                [""] + event_types,
                format_func=lambda x: "— All —" if not x else x,
                key="evt_type_filter",
            )
        with e_col3:
            evt_actor = st.selectbox(
                "Actor",
                [""] + actors,
                format_func=lambda x: "— All —" if not x else x,
                key="evt_actor_filter",
            )
        with e_col4:
            evt_page_size = st.selectbox(
                "Rows per page", [25, 50, 100, 200], index=1, key="evt_page_size"
            )

        evt_params: dict = {"page": 1, "page_size": evt_page_size}
        if evt_brief_id.strip():
            evt_params["brief_id"] = evt_brief_id.strip()
        if evt_type:
            evt_params["event_type"] = evt_type
        if evt_actor:
            evt_params["actor"] = evt_actor

        result = run_with_progress(
            "Loading audit events…",
            lambda: _get(api_base, "/audit/events", evt_params),
            estimated_seconds=5,
        )
        if result is None:
            st.error("Failed to load audit events.")
        else:
            events = result.get("events", [])
            total = result.get("total", 0)

            if not events:
                st.info("No audit events match the selected filters.")
            else:
                rows = []
                for e in events:
                    import json
                    data_str = json.dumps(e.get("event_data", {}), separators=(",", ":"))
                    rows.append({
                        "ID": e["id"],
                        "Timestamp": e["ts"],
                        "Brief ID": e["brief_id"],
                        "Event Type": e["event_type"],
                        "Actor": e["actor"],
                        "Data": data_str,
                    })
                df_e = pd.DataFrame(rows)
                _download_csv(df_e, "craftai_audit_events.csv", "⬇️ Download events (CSV)")
                st.dataframe(
                    df_e,
                    column_config={
                        **_id_column_config(),
                        "Data": st.column_config.TextColumn("Data", width="large"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(f"Showing {len(events)} of {total} total event(s)")

                with st.expander("Expand an event to see full data"):
                    if events:
                        selected_id = st.selectbox(
                            "Select event ID",
                            [e["id"] for e in events],
                            key="evt_detail_select",
                        )
                        selected = next((e for e in events if e["id"] == selected_id), None)
                        if selected:
                            st.markdown(f"**Brief ID:** `{selected['brief_id']}`")
                            st.markdown(f"**Type:** `{selected['event_type']}`")
                            st.markdown(f"**Actor:** `{selected['actor']}`")
                            st.markdown(f"**Timestamp:** `{selected['ts']}`")
                            st.json(selected.get("event_data", {}))
