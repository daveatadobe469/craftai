"""Persist main / sub-tab selection across Streamlit reruns."""

from __future__ import annotations

import streamlit as st

MAIN_TABS: list[str] = [
    "⚙️  LLM Configuration",
    "📂  Document Upload & Pipeline",
    "✍️  Content Studio",
    "📋  Audit & Observability",
]

STUDIO_TABS: list[str] = [
    "✍️ Create Brief",
    "🔍 Search Brief",
]

MAIN_TAB_LLM = MAIN_TABS[0]
MAIN_TAB_UPLOAD = MAIN_TABS[1]
MAIN_TAB_STUDIO = MAIN_TABS[2]
MAIN_TAB_AUDIT = MAIN_TABS[3]

STUDIO_TAB_CREATE = STUDIO_TABS[0]
STUDIO_TAB_SEARCH = STUDIO_TABS[1]

_MAIN_RADIO_KEY = "craft_main_nav_radio"
_STUDIO_RADIO_KEY = "craft_studio_nav_radio"


def _init_nav_state() -> None:
    st.session_state.setdefault("craft_main_tab", MAIN_TAB_LLM)
    st.session_state.setdefault("craft_studio_tab", STUDIO_TAB_CREATE)
    if st.session_state["craft_main_tab"] not in MAIN_TABS:
        st.session_state["craft_main_tab"] = MAIN_TAB_LLM
    if st.session_state["craft_studio_tab"] not in STUDIO_TABS:
        st.session_state["craft_studio_tab"] = STUDIO_TAB_CREATE


def navigate_to(main_tab: str, studio_tab: str | None = None) -> None:
    """
    Select tabs programmatically, then call st.rerun().
    Clears radio widget keys so the next render picks up craft_*_tab via index=.
    """
    if main_tab in MAIN_TABS:
        st.session_state["craft_main_tab"] = main_tab
        st.session_state.pop(_MAIN_RADIO_KEY, None)
    if studio_tab is not None and studio_tab in STUDIO_TABS:
        st.session_state["craft_studio_tab"] = studio_tab
        st.session_state.pop(_STUDIO_RADIO_KEY, None)


def render_main_nav() -> str:
    _init_nav_state()
    main_idx = MAIN_TABS.index(st.session_state["craft_main_tab"])
    choice = st.radio(
        "Main navigation",
        MAIN_TABS,
        index=main_idx,
        horizontal=True,
        label_visibility="collapsed",
        key=_MAIN_RADIO_KEY,
    )
    st.session_state["craft_main_tab"] = choice
    st.markdown("<div style='margin-bottom:14px;'></div>", unsafe_allow_html=True)
    return choice


def render_studio_nav() -> str:
    _init_nav_state()
    studio_idx = STUDIO_TABS.index(st.session_state["craft_studio_tab"])
    choice = st.radio(
        "Content Studio section",
        STUDIO_TABS,
        index=studio_idx,
        horizontal=True,
        label_visibility="collapsed",
        key=_STUDIO_RADIO_KEY,
    )
    st.session_state["craft_studio_tab"] = choice
    st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
    return choice
