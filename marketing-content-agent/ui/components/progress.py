from __future__ import annotations

import concurrent.futures
import html
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar

import streamlit as st

T = TypeVar("T")

_CSS_KEY = "_craft_progress_css_injected"
_CIRCUMFERENCE = 339.292  # 2 * pi * 54
_RING_HEIGHT = 220


def inject_progress_css() -> None:
    """Hide Streamlit's default spinner (global styles only)."""
    if st.session_state.get(_CSS_KEY):
        return
    st.session_state[_CSS_KEY] = True
    st.markdown(
        """
<style>
div[data-testid="stSpinner"],
div[data-testid="stSpinner"] > div,
.stSpinner { display: none !important; }
</style>
""",
        unsafe_allow_html=True,
    )


def render_html(container: Any, body: str, *, height: int = _RING_HEIGHT) -> None:
    """Render HTML in a Streamlit container (st.html when available)."""
    inject_progress_css()
    if hasattr(container, "html"):
        # Streamlit 1.58+: html() accepts body and width only
        container.html(body, width="stretch")
    else:
        container.markdown(body, unsafe_allow_html=True)


def ring_html(pct: int, label: str, sublabel: str = "") -> str:
    """Circular progress ring — fully inline-styled so it renders in all Streamlit versions."""
    pct = max(0, min(100, int(pct)))
    offset = _CIRCUMFERENCE * (1 - pct / 100)
    safe_label = html.escape(label)
    safe_sub = html.escape(sublabel) if sublabel else ""
    sub_block = (
        f'<div style="margin-top:6px;text-align:center;font-size:0.72rem;color:#64748b;">'
        f"{safe_sub}</div>"
        if safe_sub
        else ""
    )
    return f"""
<div style="display:flex;justify-content:center;align-items:center;
  padding:24px 16px;margin:10px 0 16px;
  background:linear-gradient(135deg,rgba(11,26,46,0.97) 0%,rgba(7,8,15,0.99) 100%);
  border:1px solid #1b3558;border-radius:16px;
  box-shadow:0 0 32px rgba(0,212,255,0.12);">
  <div style="text-align:center;">
    <div style="position:relative;width:120px;height:120px;margin:0 auto;">
      <svg viewBox="0 0 120 120" aria-label="Progress {pct} percent"
        style="width:120px;height:120px;transform:rotate(-90deg);display:block;">
        <circle cx="60" cy="60" r="54" fill="none" stroke="#1e293b" stroke-width="10"/>
        <circle cx="60" cy="60" r="54" fill="none" stroke="#00d4ff" stroke-width="10"
          stroke-linecap="round"
          stroke-dasharray="{_CIRCUMFERENCE:.3f}"
          stroke-dashoffset="{offset:.3f}"
          style="filter:drop-shadow(0 0 6px rgba(0,212,255,0.55));"/>
      </svg>
      <div style="position:absolute;top:0;left:0;width:120px;height:120px;
        display:flex;align-items:center;justify-content:center;pointer-events:none;">
        <span style="font-size:1.65rem;font-weight:900;color:#00d4ff;line-height:1;">{pct}%</span>
      </div>
    </div>
    <div style="margin-top:10px;font-size:0.84rem;font-weight:600;color:#e2e8f0;
      max-width:320px;line-height:1.35;margin-left:auto;margin-right:auto;">{safe_label}</div>
    {sub_block}
  </div>
</div>
"""


def show_ring(
    container: Any,
    pct: int,
    label: str,
    sublabel: str = "",
    *,
    height: int = _RING_HEIGHT,
) -> None:
    """Display the circular progress ring in any Streamlit placeholder/container."""
    render_html(container, ring_html(pct, label, sublabel), height=height)


class ProgressOverlay:
    """Inline circular progress indicator (replaces st.spinner)."""

    def __init__(self) -> None:
        self._ph = st.empty()

    def update(self, pct: int, label: str, sublabel: str = "") -> None:
        show_ring(self._ph, pct, label, sublabel)

    def clear(self) -> None:
        self._ph.empty()

    def finish(self, label: str = "Complete", *, auto_clear: bool = True) -> None:
        self.update(100, label, "Done")
        time.sleep(0.6)
        if auto_clear:
            self.clear()


@contextmanager
def processing(
    label: str,
    *,
    start_pct: int = 10,
    finish_label: str = "Complete",
) -> Generator[ProgressOverlay, None, None]:
    overlay = ProgressOverlay()
    overlay.update(start_pct, label)
    try:
        yield overlay
    finally:
        overlay.finish(finish_label)


def run_with_progress(
    label: str,
    func: Callable[[], T],
    *,
    estimated_seconds: float = 4.0,
    sublabel: str = "",
    keep_visible: bool = False,
) -> T:
    """Run a blocking call while animating progress toward ~92%, then 100%."""
    overlay = ProgressOverlay()
    overlay.update(8, label, sublabel)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func)
        start = time.time()
        while not future.done():
            elapsed = time.time() - start
            pct = min(92, int(8 + (elapsed / max(estimated_seconds, 0.5)) * 84))
            overlay.update(pct, label, sublabel or f"Processing… {pct}%")
            time.sleep(0.08)

        try:
            result = future.result()
        except Exception:
            overlay.clear()
            raise

    overlay.finish(label, auto_clear=not keep_visible)
    return result


def steps_to_percent(steps: list, *, total: int = 7) -> int:
    """Map pipeline step statuses to an overall completion percentage."""
    if total <= 0:
        return 0
    score = 0.0
    for step in steps:
        status = getattr(step, "status", step.get("status") if isinstance(step, dict) else "pending")
        if status == "done":
            score += 1.0
        elif status in ("running", "waiting"):
            score += 0.55
        elif status == "error":
            score += 1.0
    return min(100, int((score / total) * 100))
