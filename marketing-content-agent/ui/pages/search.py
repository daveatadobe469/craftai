from __future__ import annotations

import httpx
import streamlit as st


def render() -> None:
    st.title("🔍 Knowledge Base Search")
    st.markdown("Semantically search approved campaigns, social content, and brand guidelines.")

    api_base = st.session_state.get("api_base", "http://localhost:8000/api/v1")

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Search query",
            placeholder="e.g. premium skincare email campaign with discount offer",
        )

    with col2:
        collection = st.selectbox(
            "Collection",
            options=["approved_campaigns", "social_content", "brand_guidelines"],
            format_func=lambda x: {
                "approved_campaigns": "✅ Approved Campaigns",
                "social_content": "📱 Social Content",
                "brand_guidelines": "📋 Brand Guidelines",
            }[x],
        )

    col3, col4 = st.columns([1, 3])
    with col3:
        top_k = st.slider("Results", min_value=1, max_value=20, value=5)

    brand_filter = st.text_input(
        "Filter by Brand (optional)",
        placeholder="Leave blank for all brands",
    )
    channel_filter = st.selectbox(
        "Filter by Channel (optional)",
        options=["", "email", "linkedin", "social", "ad", "blog"],
        format_func=lambda x: "— All Channels —" if not x else x.capitalize(),
    )

    search_clicked = st.button("🔎 Search", type="primary", use_container_width=False)

    if not search_clicked:
        return

    if not query.strip():
        st.warning("Please enter a search query.")
        return

    filters: dict = {}
    if brand_filter.strip():
        filters["brand"] = brand_filter.strip()
    if channel_filter:
        filters["channel"] = channel_filter

    payload = {
        "query": query.strip(),
        "collection": collection,
        "top_k": top_k,
        "filters": filters,
    }

    with st.spinner("Searching knowledge base…"):
        try:
            resp = httpx.post(f"{api_base}/search", json=payload, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
        except httpx.ConnectError:
            st.error(f"Cannot connect to API at `{api_base}`.")
            return
        except httpx.HTTPStatusError as exc:
            st.error(f"API error {exc.response.status_code}: {exc.response.text}")
            return

    results = data.get("results", [])
    total = data.get("total", 0)

    if total == 0:
        st.info(
            "No results found. The knowledge base may be empty — "
            "run `make index` to seed it, or approve some content first."
        )
        return

    st.markdown(f"**{total} result(s)** for `{query}` in `{collection}`")
    st.divider()

    for i, result in enumerate(results, start=1):
        doc_text = result.get("document", "")
        meta = result.get("metadata", {})
        score = result.get("score", 0.0)

        with st.container():
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #d0d0d0;
                    border-radius: 8px;
                    padding: 16px;
                    margin-bottom: 12px;
                    background: #fafafa;
                ">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <strong>Result {i}</strong>
                        <span style="color:#2e7d32;font-weight:600;">Score: {score:.3f}</span>
                    </div>
                    <hr style="margin:8px 0;">
                    <p style="white-space:pre-wrap;font-family:monospace;font-size:0.85rem;">{doc_text[:600]}{"…" if len(doc_text) > 600 else ""}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            meta_display = {
                k: v for k, v in meta.items()
                if k not in ("chunk_index", "total_chunks", "doc_base_id")
            }
            if meta_display:
                with st.expander(f"Metadata — Result {i}"):
                    for key, val in meta_display.items():
                        st.text(f"{key}: {val}")
