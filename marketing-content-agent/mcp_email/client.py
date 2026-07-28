"""[mcp-email] Thin MCP client — how CraftAI's backend calls the email MCP server.

This is the "CraftAI uses an MCP server" half: the API imports these helpers and
speaks MCP (streamable-HTTP) to the server main.py started.
"""
from __future__ import annotations

import json
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from mcp_email import settings


async def _call_tool(name: str, args: dict[str, Any]) -> dict:
    """Open a short-lived MCP session, call one tool, return its parsed JSON result."""
    async with streamablehttp_client(settings.URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, args)
    text = result.content[0].text if result.content else "{}"
    return json.loads(text)


async def render_eml(brief_id: str, to: str = "") -> dict:
    """Return {subject, to, eml_base64} for local .eml download (no SMTP)."""
    return await _call_tool("render_campaign_email", {"brief_id": brief_id, "to": to})


async def render_html(brief_id: str) -> dict:
    """Return {title, html} for non-email channels to download as an .html file."""
    return await _call_tool("render_campaign_html", {"brief_id": brief_id})


async def send_email(brief_id: str, to: str = "") -> dict:
    """Render + send over SMTP; returns {status, detail, subject, to}."""
    return await _call_tool("send_campaign_email", {"brief_id": brief_id, "to": to})
