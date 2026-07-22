from __future__ import annotations

import httpx
import streamlit as st

_GROQ_MODELS_FALLBACK = [
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gemma-7b-it",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]

_EMBEDDING_MODELS_FALLBACK = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
]


def _fetch_current_config(api_base: str) -> dict:
    try:
        r = httpx.get(f"{api_base}/config", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _fetch_groq_models(api_base: str) -> list[str]:
    try:
        r = httpx.get(f"{api_base}/config/groq/models", timeout=5.0)
        r.raise_for_status()
        return r.json().get("models", _GROQ_MODELS_FALLBACK)
    except Exception:
        return _GROQ_MODELS_FALLBACK


def _fetch_embedding_models(api_base: str) -> list[str]:
    try:
        r = httpx.get(f"{api_base}/config/embedding/models", timeout=5.0)
        r.raise_for_status()
        return r.json().get("models", _EMBEDDING_MODELS_FALLBACK)
    except Exception:
        return _EMBEDDING_MODELS_FALLBACK


def _fetch_ollama_models(api_base: str, ollama_url: str) -> dict:
    try:
        r = httpx.get(
            f"{api_base}/config/ollama/models",
            params={"base_url": ollama_url},
            timeout=6.0,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"available": False, "models": [], "error": "Cannot reach API"}


def render() -> None:
    st.title("⚙️ LLM & Embedding Configuration")
    st.markdown(
        "Choose your LLM provider and embedding model. "
        "Click **Save & Apply** to persist the settings — all other tabs will use them immediately."
    )

    api_base = st.session_state.get("api_base", "http://localhost:8000/api/v1")

    # ── Load current config ───────────────────────────────────────────────────
    current = _fetch_current_config(api_base)
    if not current:
        st.warning("Cannot reach API. Showing defaults — start the backend first.")
        current = {
            "provider": "groq", "groq_model": "llama3-70b-8192",
            "groq_api_key_set": False, "ollama_base_url": "http://localhost:11434",
            "ollama_model": "llama3", "embedding_model": "all-MiniLM-L6-v2",
            "judge_threshold": 0.7, "crag_threshold": 0.5, "max_revisions": 3,
        }

    # ── Provider selection ────────────────────────────────────────────────────
    st.markdown("### 🤖 LLM Provider")

    provider = st.radio(
        "Select LLM provider",
        options=["groq", "ollama"],
        index=0 if current.get("provider", "groq") == "groq" else 1,
        format_func=lambda x: "☁️  Groq (Cloud API — fast, free tier available)" if x == "groq"
                              else "🖥️  Ollama (Local — 100% offline, no API key needed)",
        horizontal=True,
    )

    st.divider()

    # ── Groq section ──────────────────────────────────────────────────────────
    groq_api_key = current_groq_model = ""
    ollama_model = current.get("ollama_model", "llama3")
    ollama_base_url = current.get("ollama_base_url", "http://localhost:11434")

    if provider == "groq":
        st.markdown("### ☁️ Groq Configuration")

        col1, col2 = st.columns(2)
        with col1:
            api_key_display = "••••••••••••••••" if current.get("groq_api_key_set") else ""
            groq_api_key = st.text_input(
                "Groq API Key",
                value=api_key_display,
                type="password",
                placeholder="gsk_...",
                help="Get a free key at https://console.groq.com",
            )
            if groq_api_key == "••••••••••••••••":
                groq_api_key = ""

            if current.get("groq_api_key_set"):
                st.success("API key is currently set ✓")
            else:
                st.warning("API key not set")

        with col2:
            groq_models = _fetch_groq_models(api_base)
            current_groq_model = current.get("groq_model", "llama3-70b-8192")
            idx = groq_models.index(current_groq_model) if current_groq_model in groq_models else 0
            current_groq_model = st.selectbox(
                "Groq Model",
                options=groq_models,
                index=idx,
                help="llama3-70b-8192 gives best quality. llama3-8b-8192 is faster.",
            )

    # ── Ollama section ────────────────────────────────────────────────────────
    else:
        st.markdown("### 🖥️ Ollama Configuration")

        col1, col2 = st.columns(2)
        with col1:
            ollama_base_url = st.text_input(
                "Ollama Server URL",
                value=current.get("ollama_base_url", "http://localhost:11434"),
                help="Default: http://localhost:11434. Run `ollama serve` to start.",
            )

        with col2:
            st.markdown("&nbsp;")
            refresh_ollama = st.button("🔄 Refresh Model List", use_container_width=True)

        if "ollama_data" not in st.session_state or refresh_ollama:
            with st.spinner("Querying Ollama for installed models…"):
                st.session_state["ollama_data"] = _fetch_ollama_models(api_base, ollama_base_url)

        ollama_data = st.session_state.get("ollama_data", {})

        if not ollama_data.get("available"):
            err = ollama_data.get("error", "Ollama not reachable")
            st.error(f"**Ollama not available:** {err}")
            st.info(
                "To use Ollama:\n"
                "1. Install: `brew install ollama`\n"
                "2. Pull a model: `ollama pull llama3`\n"
                "3. Start server: `ollama serve`\n"
                "4. Click **Refresh Model List** above."
            )
            ollama_models = [current.get("ollama_model", "llama3")]
        else:
            raw_models = ollama_data.get("models", [])
            if raw_models:
                st.success(f"✓ Ollama running — {len(raw_models)} model(s) installed")
                model_rows = []
                for m in raw_models:
                    size = m.get("size_gb", 0)
                    family = m.get("family", "")
                    model_rows.append(f"{m['name']} ({size} GB, {family})")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.dataframe(
                        [{"Model": m["name"], "Size (GB)": m["size_gb"], "Family": m["family"]}
                         for m in raw_models],
                        use_container_width=True,
                        hide_index=True,
                    )
                ollama_models = [m["name"] for m in raw_models]
            else:
                st.warning("Ollama is running but no models are installed.")
                st.code("ollama pull llama3")
                ollama_models = ["llama3"]

        curr_idx = ollama_models.index(current.get("ollama_model", "")) \
                   if current.get("ollama_model", "") in ollama_models else 0
        ollama_model = st.selectbox(
            "Select Ollama Model",
            options=ollama_models,
            index=curr_idx,
            help="Only models already pulled via `ollama pull <name>` appear here.",
        )

    # ── Embedding model ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔢 Embedding Model (for RAG)")
    st.caption(
        "Used by ChromaDB to embed documents and queries. "
        "Changing this requires re-indexing your knowledge base (`make index`)."
    )

    emb_col1, emb_col2 = st.columns([2, 1])
    with emb_col1:
        embedding_models = _fetch_embedding_models(api_base)
        current_emb = current.get("embedding_model", "all-MiniLM-L6-v2")
        emb_idx = embedding_models.index(current_emb) if current_emb in embedding_models else 0
        embedding_model = st.selectbox(
            "Embedding Model",
            options=embedding_models,
            index=emb_idx,
            help="all-MiniLM-L6-v2 is fast and works well for English. Use multilingual variant for other languages.",
        )

    with emb_col2:
        st.metric("Current Embedding", current_emb, help="What is saved in .env right now")

    # ── Agent thresholds ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🎛️ Agent Thresholds")

    th_col1, th_col2, th_col3 = st.columns(3)
    with th_col1:
        judge_threshold = st.slider(
            "Judge Score Threshold",
            min_value=0.0, max_value=1.0,
            value=float(current.get("judge_threshold", 0.7)),
            step=0.05,
            help="Minimum LLM-as-judge score to pass compliance. Lower = more permissive.",
        )
    with th_col2:
        crag_threshold = st.slider(
            "CRAG Relevance Threshold",
            min_value=0.0, max_value=1.0,
            value=float(current.get("crag_threshold", 0.5)),
            step=0.05,
            help="Minimum relevance score to keep a retrieved chunk. Lower = more context.",
        )
    with th_col3:
        max_revisions = st.slider(
            "Max Revision Loops",
            min_value=1, max_value=5,
            value=int(current.get("max_revisions", 3)),
            step=1,
            help="How many Generate↔Comply loops before escalating to human review.",
        )

    # ── Action buttons ────────────────────────────────────────────────────────
    st.divider()
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])

    with btn_col1:
        save_clicked = st.button(
            "💾 Save & Apply Configuration",
            type="primary",
            use_container_width=True,
        )

    with btn_col2:
        test_clicked = st.button(
            "🧪 Test Connection",
            use_container_width=True,
        )

    with btn_col3:
        if st.button("🔄 Reload Current", use_container_width=True):
            st.session_state.pop("ollama_data", None)
            st.rerun()

    # ── Build payload ─────────────────────────────────────────────────────────
    payload: dict = {
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

    if test_clicked:
        _test_connection(api_base, payload)

    if save_clicked:
        _save_config(api_base, payload)


def _test_connection(api_base: str, payload: dict) -> None:
    with st.spinner("Testing LLM connection…"):
        try:
            #print(payload)
            r = httpx.post(f"{api_base}/config/test", json=payload, timeout=30.0)
            r.raise_for_status()
            result = r.json()
        except httpx.ConnectError:
            st.error("Cannot reach API. Make sure the backend is running.")
            return
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            return

    if result.get("success"):
        ms = result.get("latency_ms", "?")
        model = result.get("model", "")
        preview = result.get("response_preview", "")
        st.success(f"✅ Connection successful! Model: **{model}** · Latency: **{ms} ms**")
        if preview:
            st.caption(f'Response preview: "{preview}"')
    else:
        err = result.get("error", "Unknown error")
        st.error(f"❌ Connection failed: {err}")


def _save_config(api_base: str, payload: dict) -> None:
    with st.spinner("Saving configuration and reloading settings…"):
        try:
            r = httpx.post(f"{api_base}/config", json=payload, timeout=15.0)
            r.raise_for_status()
            saved = r.json()
        except httpx.ConnectError:
            st.error("Cannot reach API.")
            return
        except httpx.HTTPStatusError as exc:
            st.error(f"Save failed ({exc.response.status_code}): {exc.response.text}")
            return
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            return

    st.session_state["llm_provider"] = saved.get("provider")
    st.session_state["llm_model"] = (
        saved.get("groq_model") if saved.get("provider") == "groq"
        else saved.get("ollama_model")
    )
    st.session_state["embedding_model"] = saved.get("embedding_model")
    st.session_state.pop("ollama_data", None)

    st.success(
        f"✅ Configuration saved! "
        f"**Provider:** {saved['provider'].upper()} · "
        f"**Model:** {saved.get('groq_model') if saved['provider'] == 'groq' else saved.get('ollama_model')} · "
        f"**Embedding:** {saved.get('embedding_model')}"
    )
    st.info("All pipeline nodes will now use the new LLM. No restart required.")
    st.rerun()
