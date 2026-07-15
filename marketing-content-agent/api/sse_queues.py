"""Shared SSE queues so graph nodes can push events while a brief is running."""

from __future__ import annotations

import asyncio

_queues: dict[str, asyncio.Queue[str]] = {}


def register(brief_id: str, queue: asyncio.Queue[str]) -> None:
    _queues[brief_id] = queue


def unregister(brief_id: str) -> None:
    _queues.pop(brief_id, None)


def get(brief_id: str) -> asyncio.Queue[str] | None:
    return _queues.get(brief_id)


def push_sync(brief_id: str, message: str) -> None:
    """Best-effort push from sync or async graph code."""
    queue = _queues.get(brief_id)
    if queue is None:
        return
    try:
        queue.put_nowait(message)
    except asyncio.QueueFull:
        pass
