"""[mcp-email] SMTP delivery — only used by the "Send email" mode.

The ".eml download" path never touches this file, so local testing needs no SMTP
credentials at all.
"""
from __future__ import annotations

import smtplib

from mcp_email import settings


def send_smtp(eml_bytes: bytes, sender: str, to: str) -> tuple[bool, str]:
    """Send pre-built ``.eml`` bytes over SMTP. Returns (ok, human-readable detail)."""
    if not settings.smtp_configured():
        return False, "SMTP is not configured — set SMTP_HOST (and creds) or use .eml download."
    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20) as server:
            if settings.SMTP_STARTTLS:
                server.starttls()
            if settings.SMTP_USER:
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(sender, [to], eml_bytes)
        return True, f"Sent to {to} via {settings.SMTP_HOST}."
    except Exception as exc:  # noqa: BLE001 — surface any SMTP failure to the caller
        return False, f"SMTP send failed: {exc}"
