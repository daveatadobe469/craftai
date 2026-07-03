from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

LOGO_PATH = ROOT / "LOGO" / "Craft AI Logo.png"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CraftAI — Content Supply Chain",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "api_base":              "http://localhost:8000/api/v1",
    "active_brief_id":       None,
    "upload_success_doc_id": None,
    "llm_provider":          None,
    "llm_model":             None,
    "embedding_model":       None,
    "ollama_data":           None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── CSS — only custom-element classes + minimal overrides ─────────────────────
def _inject_css() -> None:
    st.markdown(
        """
<style>
/* ── Hide ALL Streamlit chrome ──────────────────────────── */
section[data-testid="stSidebar"]   { display: none !important; }
[data-testid="collapsedControl"]   { display: none !important; }
header[data-testid="stHeader"]     { display: none !important; }
[data-testid="stToolbar"]          { display: none !important; }
[data-testid="stDecoration"]       { display: none !important; }
[data-testid="stStatusWidget"]     { display: none !important; }
#MainMenu                          { display: none !important; }
footer                             { display: none !important; }
.stDeployButton                    { display: none !important; }

/* ── Zero top gap so logo sits flush ──────────────────── */
.block-container {
    padding-top: 0.4rem !important;
    padding-bottom: 1rem !important;
}
[data-testid="stMainBlockContainer"] { padding-top: 0 !important; }

/* ── Header bar ───────────────────────────────────────────── */
.ca-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 14px 24px;
    background: linear-gradient(120deg, #0b0f1e 0%, #111c30 60%, #0d1a2e 100%);
    border: 1px solid #1b3558;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 0 40px rgba(0,180,255,0.06);
}
.ca-title {
    font-size: 1.9rem;
    font-weight: 900;
    letter-spacing: 1px;
    background: linear-gradient(90deg, #00d4ff 0%, #39ff14 55%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0;
}
.ca-tagline {
    font-size: 0.71rem;
    color: #4e6a85;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 3px;
}
.ca-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #0b1525;
    border: 1px solid #1b3558;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.76rem;
    color: #8aa3b8;
    white-space: nowrap;
}
.ca-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #39ff14;
    box-shadow: 0 0 7px #39ff14;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ── Section header ───────────────────────────────────────── */
.sec-hdr {
    font-size: 0.78rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid #1b3558;
    padding-bottom: 7px;
    margin-bottom: 14px;
}

/* ── Cards ────────────────────────────────────────────────── */
.ca-card {
    background: linear-gradient(135deg, #0e1a2e 0%, #101827 100%);
    border: 1px solid #1b3558;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.ca-card-cyan   { border-left: 3px solid #00d4ff; }
.ca-card-green  { border-left: 3px solid #39ff14; }
.ca-card-orange { border-left: 3px solid #ff6b35; }
.ca-card-purple { border-left: 3px solid #9b59b6; }

/* ── Metric card ──────────────────────────────────────────── */
.ca-metric {
    background: linear-gradient(135deg, #0e1a2e 0%, #101827 100%);
    border: 1px solid #1b3558;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}
.ca-metric-val  { font-size: 2.1rem; font-weight: 900; }
.ca-metric-lbl  { font-size: 0.75rem; color: #8aa3b8; margin-top: 4px; }
.ca-metric-sub  { font-size: 0.68rem; color: #4e6a85; }

/* ── Badges ───────────────────────────────────────────────── */
.ca-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: .4px;
}
.ca-badge-green  { background: rgba(57,255,20,.12); color:#39ff14; border:1px solid #39ff14; }
.ca-badge-red    { background: rgba(255,59,59,.12);  color:#ff4444; border:1px solid #ff4444; }
.ca-badge-cyan   { background: rgba(0,212,255,.12);  color:#00d4ff; border:1px solid #00d4ff; }
.ca-badge-orange { background: rgba(255,107,53,.12); color:#ff6b35; border:1px solid #ff6b35; }

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #07080f; }
::-webkit-scrollbar-thumb { background: #1b3558; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00d4ff; }
</style>
""",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
def _render_header() -> None:
    provider = st.session_state.get("llm_provider") or "groq"
    model = st.session_state.get("llm_model") or "not configured"
    emb = st.session_state.get("embedding_model") or "MiniLM-L6"

    # Use st.columns so the logo renders reliably via st.image()
    header_l, header_r = st.columns([6, 4])

    with header_l:
        logo_col, title_col = st.columns([1, 5])
        with logo_col:
            if LOGO_PATH.exists():
                st.image(str(LOGO_PATH), width=70)
            else:
                st.markdown("# 🎯")
        with title_col:
            st.markdown(
                '<div class="ca-title">CraftAI</div>'
                '<div class="ca-tagline">Content · Review · Authoring · Flow Through AI</div>',
                unsafe_allow_html=True,
            )

    with header_r:
        brief_id = st.session_state.get("active_brief_id")
        brief_text = f"Brief: {brief_id[:8]}…" if brief_id else "No active brief"
        st.markdown(
            f"""
<div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end;align-items:center;height:100%;">
  <span class="ca-pill"><span class="ca-dot"></span> Live</span>
  <span class="ca-pill">🤖 {provider.upper()} · {model[:14]}</span>
  <span class="ca-pill">🔢 {emb.split('/')[-1][:16]}</span>
  <span class="ca-pill" style="border-color:#ff6b35;color:#ff6b35;">{brief_text}</span>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#1b3558;margin:0 0 16px 0;'>", unsafe_allow_html=True)


# ── Page module loader ────────────────────────────────────────────────────────
def _load(module_name: str):
    import importlib
    mod = importlib.import_module(f"ui.pages.{module_name}")
    importlib.reload(mod)
    return mod


# ═════════════════════════════════════════════════════════════════════════════
# TAB: LLM Configuration
# ═════════════════════════════════════════════════════════════════════════════
def _tab_llm_config() -> None:
    import httpx

    api_base = st.session_state["api_base"]

    # Fetch current config
    current: dict = {}
    try:
        r = httpx.get(f"{api_base}/config", timeout=4.0)
        r.raise_for_status()
        current = r.json()
    except Exception:
        st.warning(
            "Cannot reach the API server. Start it with `python main.py`, then refresh.",
            icon="⚠️",
        )
        current = {
            "provider": "groq", "groq_model": "llama3-70b-8192",
            "groq_api_key_set": False, "ollama_base_url": "http://localhost:11434",
            "ollama_model": "llama3", "embedding_model": "all-MiniLM-L6-v2",
            "judge_threshold": 0.7, "crag_threshold": 0.5, "max_revisions": 3,
        }

    # ── Provider cards ────────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">🤖 Provider Selection</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="ca-card ca-card-cyan">'
            '<div style="font-size:1.4rem;margin-bottom:6px;">☁️ Groq</div>'
            '<div style="color:#8aa3b8;font-size:0.84rem;">Cloud API · Ultra-fast inference · Free tier</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="ca-card ca-card-green">'
            '<div style="font-size:1.4rem;margin-bottom:6px;">🖥️ Ollama</div>'
            '<div style="color:#8aa3b8;font-size:0.84rem;">100 % local · No API key · Full privacy</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    provider = st.radio(
        "Provider",
        options=["groq", "ollama"],
        index=0 if current.get("provider", "groq") == "groq" else 1,
        format_func=lambda x: "☁️ Groq (Cloud)" if x == "groq" else "🖥️ Ollama (Local)",
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    # ── Provider-specific fields ───────────────────────────────────────────────
    groq_api_key = ""
    current_groq_model = current.get("groq_model", "llama3-70b-8192")
    ollama_model = current.get("ollama_model", "llama3")
    ollama_base_url = current.get("ollama_base_url", "http://localhost:11434")

    _GROQ_MODELS = [
        "llama3-70b-8192", "llama3-8b-8192", "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant", "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768", "gemma2-9b-it", "gemma-7b-it",
    ]

    if provider == "groq":
        st.markdown('<div class="sec-hdr">☁️ Groq Settings</div>', unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        with g1:
            raw_key = st.text_input(
                "Groq API Key",
                value="••••••••••••" if current.get("groq_api_key_set") else "",
                type="password",
                placeholder="gsk_...",
                help="Free key at console.groq.com",
            )
            if raw_key == "••••••••••••":
                groq_api_key = ""
            else:
                groq_api_key = raw_key

            if current.get("groq_api_key_set"):
                st.markdown(
                    '<span class="ca-badge ca-badge-green">✓ KEY SET</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="ca-badge ca-badge-red">✗ NOT SET</span>',
                    unsafe_allow_html=True,
                )

        with g2:
            try:
                gr = httpx.get(f"{api_base}/config/groq/models", timeout=4.0)
                groq_models = gr.json().get("models", _GROQ_MODELS)
            except Exception:
                groq_models = _GROQ_MODELS
            idx = groq_models.index(current_groq_model) if current_groq_model in groq_models else 0
            current_groq_model = st.selectbox("Groq Model", groq_models, index=idx)

    else:  # Ollama
        st.markdown('<div class="sec-hdr">🖥️ Ollama Settings</div>', unsafe_allow_html=True)
        o1, o2 = st.columns([3, 1])
        with o1:
            ollama_base_url = st.text_input(
                "Ollama Server URL",
                value=current.get("ollama_base_url", "http://localhost:11434"),
            )
        with o2:
            st.write("")
            if st.button("🔄 Fetch", use_container_width=True):
                st.session_state["ollama_data"] = None

        if not st.session_state.get("ollama_data"):
            with st.spinner("Querying Ollama…"):
                try:
                    resp = httpx.get(
                        f"{api_base}/config/ollama/models",
                        params={"base_url": ollama_base_url},
                        timeout=6.0,
                    )
                    st.session_state["ollama_data"] = resp.json()
                except Exception:
                    st.session_state["ollama_data"] = {
                        "available": False, "models": [],
                        "error": "Cannot reach API",
                    }

        od = st.session_state.get("ollama_data") or {}
        if not od.get("available"):
            st.error(f"Ollama not reachable — {od.get('error','')}")
            st.code("brew install ollama && ollama pull llama3 && ollama serve")
            ollama_models = [current.get("ollama_model", "llama3")]
        else:
            raw = od.get("models", [])
            st.success(f"Ollama running — **{len(raw)} model(s)** installed")
            if raw:
                st.dataframe(
                    [{"Model": m["name"], "Size (GB)": m["size_gb"], "Family": m["family"]}
                     for m in raw],
                    use_container_width=True,
                    hide_index=True,
                )
            ollama_models = [m["name"] for m in raw] or ["llama3"]

        oi = (
            ollama_models.index(current.get("ollama_model", ""))
            if current.get("ollama_model", "") in ollama_models else 0
        )
        ollama_model = st.selectbox("Select Model", ollama_models, index=oi)

    st.divider()

    # ── Embedding model ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">🔢 Embedding Model (RAG)</div>', unsafe_allow_html=True)
    st.caption("Changing this requires re-indexing the knowledge base: `make index`")

    _EMB = [
        "all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2",
        "paraphrase-multilingual-MiniLM-L12-v2", "multi-qa-MiniLM-L6-cos-v1",
        "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5",
    ]
    try:
        er = httpx.get(f"{api_base}/config/embedding/models", timeout=4.0)
        emb_models = er.json().get("models", _EMB)
    except Exception:
        emb_models = _EMB

    curr_emb = current.get("embedding_model", "all-MiniLM-L6-v2")
    ei = emb_models.index(curr_emb) if curr_emb in emb_models else 0
    embedding_model = st.selectbox("Embedding Model", emb_models, index=ei)

    st.divider()

    # ── Agent thresholds ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">🎛️ Agent Thresholds</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    with t1:
        judge_threshold = st.slider(
            "Judge Score Threshold", 0.0, 1.0,
            float(current.get("judge_threshold", 0.7)), 0.05,
        )
        st.caption("Min score to pass compliance gate")
    with t2:
        crag_threshold = st.slider(
            "CRAG Relevance Threshold", 0.0, 1.0,
            float(current.get("crag_threshold", 0.5)), 0.05,
        )
        st.caption("Min score to keep a RAG chunk")
    with t3:
        max_revisions = st.slider(
            "Max Revision Loops", 1, 5,
            int(current.get("max_revisions", 3)), 1,
        )
        st.caption("Generate ↔ Comply loops before escalating")

    st.divider()

    # ── Save / Test ───────────────────────────────────────────────────────────
    payload = {
        "provider": provider,
        "groq_api_key": groq_api_key or None,
        "groq_model": current_groq_model if provider == "groq" else current.get("groq_model", "llama3-70b-8192"),
        "ollama_base_url": ollama_base_url,
        "ollama_model": ollama_model if provider == "ollama" else current.get("ollama_model", "llama3"),
        "embedding_model": embedding_model,
        "judge_threshold": judge_threshold,
        "crag_threshold": crag_threshold,
        "max_revisions": max_revisions,
    }

    b1, b2 = st.columns(2)
    with b1:
        if st.button("🧪 Test Connection", use_container_width=True):
            with st.spinner("Testing LLM connection…"):
                try:
                    tr = httpx.post(f"{api_base}/config/test", json=payload, timeout=30.0)
                    res = tr.json()
                    if res.get("success"):
                        st.success(
                            f"Connected — **{res.get('model')}** · {res.get('latency_ms')} ms\n\n"
                            f"> {res.get('response_preview', '')}"
                        )
                    else:
                        st.error(f"Connection failed: {res.get('error')}")
                except Exception as exc:
                    st.error(f"Request failed: {exc}")

    with b2:
        if st.button("💾 Save & Apply", use_container_width=True, type="primary"):
            with st.spinner("Saving configuration…"):
                try:
                    sr = httpx.post(f"{api_base}/config", json=payload, timeout=15.0)
                    sr.raise_for_status()
                    saved = sr.json()
                    _prov = saved.get("provider", "")
                    _mdl  = saved.get("groq_model") if _prov == "groq" else saved.get("ollama_model")
                    _emb  = saved.get("embedding_model", "")
                    st.session_state["llm_provider"]   = _prov
                    st.session_state["llm_model"]      = _mdl
                    st.session_state["embedding_model"] = _emb
                    st.session_state["ollama_data"]    = None
                    st.success(
                        f"✅ **Configuration saved successfully!**\n\n"
                        f"- Provider: **{_prov.upper()}**\n"
                        f"- Model: **{_mdl}**\n"
                        f"- Embedding: **{_emb}**\n\n"
                        f"All agents will now use this LLM."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Save failed: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB: Document Upload & Pipeline
# ═════════════════════════════════════════════════════════════════════════════
_COLLS = {
    "brand_guidelines":   ("📋 Brand Guidelines",   "#00d4ff"),
    "approved_campaigns": ("✅ Approved Campaigns", "#39ff14"),
    "social_content":     ("📱 Social Content",     "#ff6b35"),
}


def _tab_upload() -> None:
    import httpx

    api_base = st.session_state["api_base"]

    # ── KB stats ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">📊 Knowledge Base Status</div>', unsafe_allow_html=True)

    stat_cols = st.columns(3)
    for col, (coll_key, (label, color)) in zip(stat_cols, _COLLS.items()):
        count = 0
        try:
            r = httpx.get(f"{api_base}/ingest/{coll_key}", timeout=4.0)
            count = r.json().get("total_documents", 0)
        except Exception:
            pass
        with col:
            st.markdown(
                f'<div class="ca-metric" style="border-left:3px solid {color};">'
                f'<div class="ca-metric-val" style="color:{color};">{count}</div>'
                f'<div class="ca-metric-lbl">{label}</div>'
                f'<div class="ca-metric-sub">chunks indexed</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Upload form + Semantic search ─────────────────────────────────────────
    up_left, up_right = st.columns([3, 2])

    with up_left:
        st.markdown('<div class="sec-hdr">📤 Upload Document</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop file here",
            type=["txt", "md", "csv", "json", "pdf", "docx"],
            label_visibility="collapsed",
        )
        fc1, fc2 = st.columns(2)
        with fc1:
            collection = st.selectbox(
                "Target Collection",
                list(_COLLS.keys()),
                format_func=lambda k: _COLLS[k][0],
            )
        with fc2:
            brand_tag = st.text_input("Brand Tag", placeholder="e.g. GlowBrand")
        fc3, fc4 = st.columns(2)
        with fc3:
            channel_tag = st.selectbox(
                "Channel Tag",
                ["", "email", "linkedin", "social", "ad", "blog"],
                format_func=lambda x: "— All —" if not x else x.capitalize(),
            )
        with fc4:
            chunk_size = st.select_slider(
                "Chunk Size (tokens)",
                options=[128, 256, 512, 768, 1024, 2048],
                value=512,
            )
        if st.button(
            "🚀 Embed & Index",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        ):
            _do_upload(api_base, uploaded_file, collection, brand_tag, channel_tag, chunk_size)

    with up_right:
        st.markdown('<div class="sec-hdr">🔍 Semantic Search</div>', unsafe_allow_html=True)
        query = st.text_area(
            "Query",
            placeholder="e.g. premium skincare email campaign…",
            height=90,
            label_visibility="collapsed",
        )
        search_coll = st.selectbox(
            "Search in",
            list(_COLLS.keys()),
            format_func=lambda k: _COLLS[k][0],
            key="search_coll",
        )
        top_k = st.slider("Top K results", 1, 10, 5)
        if st.button("🔍 Search", type="primary", use_container_width=True, disabled=not query.strip()):
            with st.spinner("Searching knowledge base…"):
                try:
                    sr = httpx.post(
                        f"{api_base}/search",
                        json={"query": query.strip(), "collection": search_coll, "top_k": top_k, "filters": {}},
                        timeout=15.0,
                    )
                    results = sr.json().get("results", [])
                    if not results:
                        st.info("No results. Upload documents or run `make index`.")
                    for i, res in enumerate(results, 1):
                        score = res.get("score", 0)
                        doc = res.get("document", "")
                        color = _COLLS.get(search_coll, ("", "#00d4ff"))[1]
                        st.markdown(
                            f'<div class="ca-card" style="border-left:3px solid {color};">'
                            f'<div style="display:flex;justify-content:space-between;">'
                            f'<span style="color:#8aa3b8;font-size:.78rem;">Result {i}</span>'
                            f'<span style="color:{color};font-weight:700;">{score:.3f}</span>'
                            f'</div>'
                            f'<div style="font-size:.82rem;color:#b0bec5;margin-top:8px;white-space:pre-wrap;">'
                            f'{doc[:300]}{"…" if len(doc) > 300 else ""}'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
                except Exception as exc:
                    st.error(f"Search failed: {exc}")

    st.divider()

    # ── Collection browser ────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">📚 Collection Browser</div>', unsafe_allow_html=True)
    bc1, bc2 = st.columns([3, 1])
    with bc1:
        browse_coll = st.selectbox(
            "Browse",
            list(_COLLS.keys()),
            format_func=lambda k: _COLLS[k][0],
            key="browse_coll",
        )
    with bc2:
        st.write("")
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    try:
        br = httpx.get(f"{api_base}/ingest/{browse_coll}", timeout=4.0)
        bdata = br.json()
        total = bdata.get("total_documents", 0)
        sample = bdata.get("sample", [])
        color = _COLLS.get(browse_coll, ("", "#00d4ff"))[1]
        st.markdown(
            f'<span class="ca-badge ca-badge-cyan">{total} CHUNKS</span>',
            unsafe_allow_html=True,
        )
        st.write("")
        if sample:
            for i, item in enumerate(sample, 1):
                prev = item.get("preview", "")
                src = item.get("metadata", {}).get("source_filename", "seeded")
                st.markdown(
                    f'<div class="ca-card" style="border-left:3px solid {color};">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
                    f'<span style="color:{color};font-weight:700;font-size:.78rem;">CHUNK {i}</span>'
                    f'<span style="color:#4e6a85;font-size:.72rem;">{src}</span>'
                    f'</div>'
                    f'<div style="font-size:.8rem;color:#8aa3b8;">{prev}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        elif total == 0:
            st.info("Empty collection. Upload a document above or run `make index`.")
    except Exception:
        st.warning("Cannot reach API — start the backend with `python main.py`")


def _do_upload(api_base, uploaded_file, collection, brand_tag, channel_tag, chunk_size):
    import httpx

    file_bytes = uploaded_file.read()
    filename = uploaded_file.name
    with st.spinner(f"Embedding '{filename}'…"):
        try:
            r = httpx.post(
                f"{api_base}/ingest",
                files={"file": (filename, file_bytes, "application/octet-stream")},
                data={
                    "collection": collection,
                    "brand": brand_tag.strip(),
                    "channel": channel_tag,
                    "chunk_size": str(chunk_size),
                    "chunk_overlap": str(chunk_size // 8),
                },
                timeout=120.0,
            )
            r.raise_for_status()
            result = r.json()
            st.success(
                f"Embedded **{result.get('chunks')} chunks** from `{filename}` → `{collection}`"
            )
            st.code(f"Document ID: {result.get('doc_id')}")
            st.session_state["upload_success_doc_id"] = result.get("doc_id")
            st.rerun()
        except httpx.ConnectError:
            st.error("Cannot connect to API.")
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:
                detail = exc.response.text
            st.error(f"Upload failed: {detail}")
        except Exception as exc:
            st.error(f"Error: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB: Content Studio
# ═════════════════════════════════════════════════════════════════════════════
def _tab_content_studio() -> None:
    studio_tabs = st.tabs(["✍️ Create Brief", "👁️ Review Draft"])
    with studio_tabs[0]:
        _load("creator").render()
    with studio_tabs[1]:
        _load("reviewer").render()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    _inject_css()
    _render_header()

    # API URL quick override
    with st.expander("🔧 API Server URL", expanded=False):
        new_api = st.text_input(
            "Base URL",
            value=st.session_state["api_base"],
            label_visibility="collapsed",
            placeholder="http://localhost:8000/api/v1",
        )
        if new_api.rstrip("/") != st.session_state["api_base"]:
            st.session_state["api_base"] = new_api.rstrip("/")
            st.rerun()

    # Top-level tabs
    tabs = st.tabs([
        "⚙️  LLM Configuration",
        "📂  Document Upload & Pipeline",
        "✍️  Content Studio",
        "📋  Audit & Observability",
    ])

    with tabs[0]:
        _tab_llm_config()

    with tabs[1]:
        _tab_upload()

    with tabs[2]:
        _tab_content_studio()

    with tabs[3]:
        _load("audit").render()


if __name__ == "__main__" or True:
    main()
