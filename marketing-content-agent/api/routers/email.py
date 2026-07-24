"""[mcp-email] API endpoint that speaks MCP to the email server.

Thin adapter: the UI posts here on approval; this handler is the MCP *client*
(via mcp_email.client) that renders or sends the campaign email. Remove the whole
feature by deleting this file, its include_router line, and the mcp_email/ folder.
"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mcp_email import client

router = APIRouter()


class EmailPayload(BaseModel):
    mode: Literal["eml", "send", "html"] = Field(
        default="eml", description="Download .eml, send via SMTP, or download campaign HTML"
    )
    to: Optional[str] = Field(default=None, description="Recipient; falls back to the module default")


@router.post("/email/{brief_id}")
async def post_email(brief_id: str, payload: EmailPayload) -> dict:
    """Render (.eml / .html) or send the campaign for a brief, via the email MCP server."""
    to = payload.to or ""
    try:
        if payload.mode == "send":
            result = await client.send_email(brief_id, to)
        elif payload.mode == "html":
            result = await client.render_html(brief_id)
        else:
            result = await client.render_eml(brief_id, to)
    except Exception as exc:  # noqa: BLE001 — MCP server down / unreachable
        raise HTTPException(
            status_code=503,
            detail=f"Email MCP server unreachable: {exc}",
        ) from exc

    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result
