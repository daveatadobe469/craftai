from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from db.sqlite import update_brief_status
from graph.state import AgentState

if TYPE_CHECKING:
    pass

_GATE_TIMEOUT_SECONDS: int = 30 * 60

_events: dict[str, asyncio.Event] = {}
_decisions: dict[str, dict[str, Any]] = {}


def _ensure_event(brief_id: str) -> asyncio.Event:
    if brief_id not in _events:
        _events[brief_id] = asyncio.Event()
    return _events[brief_id]


def set_decision(brief_id: str, decision: str, edits: str | None = None, reviewer: str = "human") -> None:
    """
    Called externally (from the API decision endpoint) to unblock the gate.
    Stores the decision payload and sets the asyncio.Event.
    """
    _decisions[brief_id] = {
        "human_decision": decision,
        "human_edits": edits,
        "reviewed_by": reviewer,
        "reviewed_at": datetime.now(timezone.utc),
    }
    event = _ensure_event(brief_id)
    event.set()


def route(state: AgentState) -> str:
    """
    Conditional edge router after human_gate_node.
    Returns "approved" or "rejected".
    """
    decision = state.get("human_decision")
    if decision in ("approved", "edited"):
        return "approved"
    return "rejected"


async def human_gate_node(state: AgentState) -> AgentState:
    """
    Stage 4 — Human Gate Node
    Suspends the graph until a human decision arrives via set_decision().
    Waits up to 30 minutes, then auto-escalates as 'rejected' with a timeout note.
    """
    errors: list[str] = list(state.get("errors") or [])
    sse_events: list[str] = list(state.get("sse_events") or [])

    try:
        brief_id = state["brief_id"]
        draft = state.get("draft", "")
        channel = state["channel"]
        judge_score = state.get("judge_score", 0.0)
        compliance_pass = state.get("compliance_pass", False)

        sse_events.append(
            f"[HumanGate] Awaiting review. "
            f"Compliance: {'PASS' if compliance_pass else 'FAIL'}. "
            f"Judge score: {judge_score:.2f}. "
            f"Timeout: 30 min."
        )
        sse_events.append("__human_action_required__")

        event = _ensure_event(brief_id)
        event.clear()

        _decisions.pop(brief_id, None)

        try:
            await asyncio.wait_for(event.wait(), timeout=_GATE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            sse_events.append("[HumanGate] Timeout — auto-rejecting after 30 minutes.")
            await asyncio.get_event_loop().run_in_executor(
                None, update_brief_status, brief_id, "rejected"
            )
            return {
                **state,
                "human_decision": "rejected",
                "human_edits": None,
                "reviewed_by": "system-timeout",
                "reviewed_at": datetime.now(timezone.utc),
                "errors": errors,
                "sse_events": sse_events,
            }

        payload = _decisions.pop(brief_id, {})
        human_decision = payload.get("human_decision", "rejected")
        human_edits = payload.get("human_edits")
        reviewed_by = payload.get("reviewed_by", "unknown")
        reviewed_at = payload.get("reviewed_at", datetime.now(timezone.utc))

        _events.pop(brief_id, None)

        if human_decision == "rejected":
            await asyncio.get_event_loop().run_in_executor(
                None, update_brief_status, brief_id, "rejected"
            )

        sse_events.append(
            f"[HumanGate] Decision received: {human_decision} by {reviewed_by}."
        )

        return {
            **state,
            "human_decision": human_decision,
            "human_edits": human_edits,
            "reviewed_by": reviewed_by,
            "reviewed_at": reviewed_at,
            "errors": errors,
            "sse_events": sse_events,
        }

    except Exception as exc:
        errors.append(f"HumanGate error: {exc}")
        sse_events.append(f"[HumanGate] ERROR: {exc}")
        return {
            **state,
            "human_decision": "rejected",
            "errors": errors,
            "sse_events": sse_events,
        }
