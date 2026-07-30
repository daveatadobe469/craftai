"""[mcp-email] The email MCP server (streamable-HTTP).

Exposes campaign-email delivery as MCP tools so CraftAI's backend can call it as
an MCP *client*. Started as a subprocess by main.py; run directly with:

    python -m mcp_email.server

Tools:
    render_campaign_email(brief_id, to)  -> {subject, to, eml_base64}   (no SMTP)
    send_campaign_email(brief_id, to)    -> {status, detail, subject, to}
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from db.sqlite import get_brief, get_latest_draft
from mcp_email import composer, sender, settings

mcp = FastMCP("craftai-email", host=settings.HOST, port=settings.PORT, stateless_http=True)


def _meta(draft: dict[str, Any]) -> dict:
    """Draft metadata may be a dict or a JSON string — normalise to a dict."""
    raw = draft.get("metadata")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}


def _load(brief_id: str) -> dict | None:
    """Gather everything the email needs from the brief + its latest draft.

    Email-channel drafts carry clean structured fields (subject/body/cta) in the
    draft metadata; prefer those over the raw draft text, which mashes them into
    one "Subject: … Body: …" blob. Human edits, when present, win over both.
    """
    brief = get_brief(brief_id)
    if not brief:
        return None
    draft = get_latest_draft(brief_id) or {}
    meta = _meta(draft)
    edits = draft.get("human_edits")
    return {
        "brand": brief.get("brand", "Your Brand"),
        "channel": brief.get("channel", "campaign"),
        "subject": meta.get("subject"),
        "body": edits or meta.get("body") or draft.get("content") or "",
        "cta": meta.get("cta"),
        "image_url": meta.get("image_url"),
    }


def _render(brief_id: str) -> dict | None:
    """Build the campaign title + branded HTML for a brief. None if the brief is missing."""
    data = _load(brief_id)
    if data is None:
        return None
    title = data["subject"] or composer.subject_line(data["brand"], data["channel"], data["body"])
    html = composer.render_html(
        data["brand"], data["channel"], data["body"], data["image_url"], data["cta"]
    )
    return {"title": title, "html": html}


def _compose(brief_id: str, to: str) -> dict | None:
    """Build subject + HTML + .eml bytes for a brief. None if the brief is missing."""
    rendered = _render(brief_id)
    if rendered is None:
        return None
    to = to or settings.DEFAULT_TO
    eml = composer.build_eml(rendered["title"], settings.SENDER, to, rendered["html"])
    return {"subject": rendered["title"], "to": to, "eml": eml}


@mcp.tool()
def render_campaign_email(brief_id: str, to: str = "") -> str:
    """Render the campaign email for a brief and return it as .eml bytes (base64). No sending."""
    built = _compose(brief_id, to)
    if built is None:
        return json.dumps({"error": f"No brief found for id '{brief_id}'."})
    return json.dumps({
        "subject": built["subject"],
        "to": built["to"],
        "eml_base64": base64.b64encode(built["eml"]).decode("ascii"),
    })


@mcp.tool()
def render_campaign_html(brief_id: str) -> str:
    """Render the campaign as a standalone HTML document (for non-email channels). No sending."""
    rendered = _render(brief_id)
    if rendered is None:
        return json.dumps({"error": f"No brief found for id '{brief_id}'."})
    return json.dumps({"title": rendered["title"], "html": rendered["html"]})


@mcp.tool()
async def send_campaign_email(brief_id: str, to: str = "") -> str:
    """Render and send the campaign email for a brief over SMTP."""
    built = _compose(brief_id, to)
    if built is None:
        return json.dumps({"status": "error", "detail": f"No brief found for id '{brief_id}'."})
    # send_smtp is a blocking smtplib call (up to the 20s connect/banner timeout
    # on a slow or firewalled network) — run it off-thread so it doesn't stall
    # this server's single event loop, which would otherwise freeze every other
    # in-flight MCP tool call (render/send, for this brief or any other) for the
    # same duration.
    ok, detail = await asyncio.to_thread(
        sender.send_smtp, built["eml"], settings.SENDER, built["to"]
    )
    return json.dumps({
        "status": "sent" if ok else "error",
        "detail": detail,
        "subject": built["subject"],
        "to": built["to"],
    })


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
