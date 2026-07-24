"""[mcp-email] Removable email-over-MCP module.

CraftAI demonstrates the MCP *client* role here: the backend calls a separate
email MCP server (started by main.py) to render/send a campaign email when a
reviewer approves a brief.

To remove the whole feature: delete this folder and undo the four clearly-marked
`[mcp-email]` hooks in main.py, api/main.py, and ui/pages/reviewer.py.
"""
