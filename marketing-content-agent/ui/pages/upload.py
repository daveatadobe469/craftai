from __future__ import annotations

import httpx
import streamlit as st

_COLLECTION_LABELS = {
    "brand_guidelines": "📋 Brand Guidelines",
    "approved_campaigns": "✅ Approved Campaigns",
    "social_content": "📱 Social Content",
}

_COLLECTION_HELP = {
    "brand_guidelines": (
        "Tone-of-voice rules, logo usage, restricted words, channel policies. "
        "Used by the Compliance node and the LLM-as-judge."
    ),
    "approved_campaigns": (
        "Past approved marketing copy (emails, ads, blog posts, LinkedIn). "
        "Used by the Generator node via HyDE + CRAG retrieval."
    ),
    "social_content": (
        "Approved social media posts, tweets, captions. "
        "Used specifically when generating social channel content."
    ),
}

_SUPPORTED_FORMATS = ["txt", "md", "csv", "json", "pdf", "docx"]


def _get_collection_stats(api_base: str, collection: str) -> dict | None:
    try:
        resp = httpx.get(f"{api_base}/ingest/{collection}", timeout=8.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def render() -> None:
    st.title("📂 Upload Documents for RAG")
    st.markdown(
        "Upload brand guidelines, past campaigns, or social content. "
        "Documents are **chunked → embedded → stored in ChromaDB** and immediately "
        "available to the Agentic RAG retrieval pipeline (HyDE + CRAG)."
    )

    api_base = st.session_state.get("api_base", "http://localhost:8000/api/v1")

    # ── Collection stats banner ───────────────────────────────────────────────
    st.markdown("### Knowledge Base Status")
    cols = st.columns(3)
    for col, (coll_key, coll_label) in zip(cols, _COLLECTION_LABELS.items()):
        stats = _get_collection_stats(api_base, coll_key)
        with col:
            if stats is not None:
                count = stats.get("total_documents", 0)
                col.metric(coll_label, f"{count} chunks", help=_COLLECTION_HELP[coll_key])
            else:
                col.metric(coll_label, "—", help="Could not reach API")

    st.divider()

    # ── Upload form ───────────────────────────────────────────────────────────
    st.markdown("### Upload a Document")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=_SUPPORTED_FORMATS,
            help=f"Supported: {', '.join(f'.{ext}' for ext in _SUPPORTED_FORMATS)} · Max 10 MB",
        )

    with col_right:
        collection = st.selectbox(
            "Target Collection *",
            options=list(_COLLECTION_LABELS.keys()),
            format_func=lambda k: _COLLECTION_LABELS[k],
            help="Which knowledge base to add this document to.",
        )
        st.caption(_COLLECTION_HELP[collection])

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        brand_tag = st.text_input(
            "Brand Tag",
            placeholder="e.g. GlowBrand",
            help="Optional: tag chunks by brand for filtered retrieval.",
        )
    with col_meta2:
        channel_tag = st.selectbox(
            "Channel Tag",
            options=["", "email", "linkedin", "social", "ad", "blog"],
            format_func=lambda x: "— All Channels —" if not x else x.capitalize(),
            help="Optional: restrict this document to a specific channel.",
        )

    with st.expander("Advanced chunking options"):
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=128,
            max_value=2048,
            value=512,
            step=64,
            help="Larger chunks = more context per retrieval hit. Smaller = more precise.",
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=256,
            value=64,
            step=16,
            help="Overlap between consecutive chunks to avoid losing context at boundaries.",
        )

    upload_clicked = st.button(
        "🚀 Upload & Embed",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    )

    if upload_clicked and uploaded_file is not None:
        _do_upload(
            api_base=api_base,
            uploaded_file=uploaded_file,
            collection=collection,
            brand_tag=brand_tag,
            channel_tag=channel_tag,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ── Collection browser ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Browse Collection Contents")

    browse_collection = st.selectbox(
        "Select collection to browse",
        options=list(_COLLECTION_LABELS.keys()),
        format_func=lambda k: _COLLECTION_LABELS[k],
        key="browse_collection_select",
    )

    if st.button("🔄 Refresh", key="refresh_browse"):
        st.rerun()

    stats = _get_collection_stats(api_base, browse_collection)
    if stats is None:
        st.error("Cannot reach API. Make sure the backend is running.")
        return

    total = stats.get("total_documents", 0)
    sample = stats.get("sample", [])

    if total == 0:
        st.info(
            f"The **{_COLLECTION_LABELS[browse_collection]}** collection is empty. "
            "Upload a document above or run `make index` to seed it."
        )
        return

    st.markdown(f"**{total} total chunks** in `{browse_collection}`")

    if sample:
        st.markdown("**Sample chunks (first 5):**")
        for i, item in enumerate(sample, start=1):
            preview = item.get("preview", "")
            meta = item.get("metadata", {})
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 12px 16px;
                        margin-bottom: 10px;
                        background: #f9f9f9;
                    ">
                        <strong>Chunk {i}</strong>
                        <span style="float:right;font-size:0.8rem;color:#888;">
                            {meta.get('source_filename', meta.get('type', 'seeded'))}
                        </span>
                        <hr style="margin:6px 0;">
                        <code style="font-size:0.82rem;white-space:pre-wrap;">{preview}</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                meta_display = {
                    k: v for k, v in meta.items()
                    if k not in ("chunk_index", "total_chunks", "doc_base_id")
                }
                if meta_display:
                    with st.expander(f"Metadata — Chunk {i}"):
                        for k, v in meta_display.items():
                            st.text(f"{k}: {v}")

    # ── Delete section ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Delete a Document")
    st.caption(
        "Enter the Document ID (returned after upload) to remove all its chunks from a collection."
    )

    del_col1, del_col2 = st.columns([3, 1])
    with del_col1:
        delete_doc_id = st.text_input(
            "Document ID to delete",
            placeholder="e.g. 3fa85f64-5717-4562-b3fc-2c963f66afa6",
            key="delete_doc_id_input",
        )
    with del_col2:
        delete_collection = st.selectbox(
            "From collection",
            options=list(_COLLECTION_LABELS.keys()),
            format_func=lambda k: _COLLECTION_LABELS[k],
            key="delete_collection_select",
        )

    if st.button("🗑️ Delete Document", type="secondary", disabled=not delete_doc_id.strip()):
        _do_delete(api_base, delete_collection, delete_doc_id.strip())


def _do_upload(
    api_base: str,
    uploaded_file,
    collection: str,
    brand_tag: str,
    channel_tag: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name

    with st.spinner(f"Embedding '{filename}' into `{collection}`…"):
        try:
            resp = httpx.post(
                f"{api_base}/ingest",
                files={"file": (filename, file_bytes, "application/octet-stream")},
                data={
                    "collection": collection,
                    "brand": brand_tag.strip(),
                    "channel": channel_tag,
                    "chunk_size": str(chunk_size),
                    "chunk_overlap": str(chunk_overlap),
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            result = resp.json()
        except httpx.ConnectError:
            st.error(
                f"Cannot connect to API at `{api_base}`. "
                "Ensure the FastAPI backend is running (`make api`)."
            )
            return
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:
                detail = exc.response.text
            st.error(f"Upload failed ({exc.response.status_code}): {detail}")
            return
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            return

    doc_id = result.get("doc_id", "—")
    chunks = result.get("chunks", 0)
    message = result.get("message", "Upload complete.")

    st.success(f"✅ {message}")

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Chunks Created", chunks)
    info_col2.metric("Collection", _COLLECTION_LABELS.get(collection, collection))
    info_col3.metric("Chunk Size", f"{chunk_size} chars")

    st.code(f"Document ID: {doc_id}", language=None)
    st.caption(
        "Save this Document ID if you want to delete this document later. "
        "The chunks are now live in the RAG pipeline — submit a brief to see them retrieved."
    )
    st.rerun()


def _do_delete(api_base: str, collection: str, doc_id: str) -> None:
    with st.spinner(f"Deleting `{doc_id}` from `{collection}`…"):
        try:
            resp = httpx.delete(
                f"{api_base}/ingest/{collection}/{doc_id}",
                timeout=15.0,
            )
            resp.raise_for_status()
            result = resp.json()
            deleted = result.get("chunks_deleted", 0)
            st.success(f"✅ Deleted {deleted} chunk(s) for document `{doc_id}`.")
            st.rerun()
        except httpx.ConnectError:
            st.error("Cannot connect to API.")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                st.warning(f"Document `{doc_id}` not found in `{collection}`.")
            else:
                st.error(f"Delete failed ({exc.response.status_code}): {exc.response.text}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
