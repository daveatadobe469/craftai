from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

router = APIRouter()


@router.get("/stream/{brief_id}")
async def stream_events(brief_id: str, request: Request) -> EventSourceResponse:
    """
    Server-Sent Events stream for a brief_id.
    Yields agent trace messages as they are produced by graph nodes.
    """
    queues: dict[str, asyncio.Queue] = request.app.state.queues

    if brief_id not in queues:
        raise HTTPException(
            status_code=404,
            detail=f"No active stream for brief_id '{brief_id}'. Submit a brief first.",
        )

    queue = queues[brief_id]

    async def event_generator():
        tasks: dict[str, asyncio.Task] = request.app.state.tasks
        try:
            while True:
                if await request.is_disconnected():
                    break

                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield {"event": "heartbeat", "data": "ping"}
                    continue

                if message == "__done__":
                    yield {"event": "done", "data": json.dumps({"brief_id": brief_id})}
                    break
                elif message.startswith("__error__:"):
                    error_text = message[len("__error__:"):]
                    yield {"event": "error", "data": json.dumps({"error": error_text})}
                    break
                elif message == "__human_action_required__":
                    yield {
                        "event": "human_action_required",
                        "data": json.dumps({"brief_id": brief_id}),
                    }
                elif message == "__pipeline_complete__":
                    yield {
                        "event": "pipeline_complete",
                        "data": json.dumps({"brief_id": brief_id}),
                    }
                else:
                    yield {"event": "trace", "data": json.dumps({"message": message})}

        except asyncio.CancelledError:
            pass
        finally:
            task = tasks.get(brief_id)
            if task is None or task.done():
                queues.pop(brief_id, None)

    return EventSourceResponse(event_generator())
