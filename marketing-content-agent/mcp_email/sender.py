"""[mcp-email] SMTP delivery — only used by the "Send email" mode.

The ".eml download" path never touches this file, so local testing needs no SMTP
credentials at all.
"""
from __future__ import annotations

import smtplib
import socket

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
    except (TimeoutError, socket.timeout, smtplib.SMTPServerDisconnected) as exc:
        # The TCP handshake to SMTP_HOST/SMTP_PORT can succeed instantly while a
        # transparent firewall/proxy silently drops the actual SMTP traffic — the
        # client then just waits for a banner/response that never arrives until
        # the socket timeout fires. That looks identical to a dead host, so say
        # so explicitly instead of leaking a bare "timed out" that reads like a
        # credentials problem.
        return False, (
            f"SMTP send failed: no response from {settings.SMTP_HOST}:{settings.SMTP_PORT} "
            f"({exc}). The connection often completes at the TCP level even when a "
            "network/firewall (VPN, corporate proxy) is silently blocking SMTP traffic — "
            "confirm outbound SMTP is actually allowed from this network, or switch to an "
            "HTTPS-based email provider (SendGrid/Mailgun/Resend/Gmail API), which is far "
            "less likely to be blocked than raw SMTP."
        )
    except Exception as exc:  # noqa: BLE001 — surface any other SMTP failure to the caller
        return False, f"SMTP send failed: {exc}"
