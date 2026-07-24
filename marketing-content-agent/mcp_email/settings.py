"""[mcp-email] Self-contained settings for the removable email-MCP module.

Reads straight from the environment so the whole feature lives inside this one
folder — delete `mcp_email/` and nothing dangles in the central config.
"""
from __future__ import annotations

import os


def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# Whether main.py should start the email-MCP server subprocess at all.
ENABLED: bool = _flag("EMAIL_MCP_ENABLED", True)

# Where the streamable-HTTP MCP server binds / is reached.
HOST: str = os.environ.get("EMAIL_MCP_HOST", "127.0.0.1")
PORT: int = int(os.environ.get("EMAIL_MCP_PORT", "8765"))
PATH: str = os.environ.get("EMAIL_MCP_PATH", "/mcp")
URL: str = os.environ.get("EMAIL_MCP_URL", f"http://{HOST}:{PORT}{PATH}")

# Message envelope defaults.
SENDER: str = os.environ.get("EMAIL_FROM", "CraftAI <craftai@demo.local>")
DEFAULT_TO: str = os.environ.get("EMAIL_DEFAULT_TO", "reviewer@brand.com")

# SMTP — only needed for the "Send email" mode. ".eml download" needs none of this.
SMTP_HOST: str = os.environ.get("SMTP_HOST", "")
SMTP_PORT: int = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER: str = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD: str = os.environ.get("SMTP_PASSWORD", "")
SMTP_STARTTLS: bool = _flag("SMTP_STARTTLS", True)


def smtp_configured() -> bool:
    return bool(SMTP_HOST)
