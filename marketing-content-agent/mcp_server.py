"""[mcp] CraftAI MCP server — read-only, indicative.

Exposes CraftAI's marketing knowledge base and campaign records as tools over the
Model Context Protocol (MCP), so any MCP client — Claude Desktop, an IDE, or
another agent — can query CraftAI through the open standard. This is the
curriculum's "connect external tools and services via MCP" integration.

It reuses the existing RAG + database code and is strictly read-only: it never
runs the pipeline or writes data, so it is safe to expose alongside the app.

Run over stdio (the transport Claude Desktop and most MCP clients use):

    python mcp_server.py

Register it with an MCP client, e.g. Claude Desktop's config:

    {
      "mcpServers": {
        "craftai": {
          "command": "/ABS/PATH/venv/bin/python",
          "args": ["/ABS/PATH/marketing-content-agent/mcp_server.py"]
        }
      }
    }
"""
from __future__ import annotations

import json
import sqlite3

from typing import Any

from mcp.server.fastmcp import FastMCP

from config import settings
from db.sqlite import get_brief, get_latest_draft, get_personas
from rag import chroma_client as cc
from rag import embedder

mcp = FastMCP("craftai")

# Friendly collection aliases → the ChromaDB collection names.
_COLLECTIONS = {
    "campaigns": cc.APPROVED_CAMPAIGNS,   # approved past campaigns
    "social": cc.SOCIAL_CONTENT,          # social-media samples
    "guidelines": cc.BRAND_GUIDELINES,    # brand tone / rules
}


def _meta(draft: dict[str, Any]) -> dict:
    """Draft metadata may come back as a dict or a JSON string — normalise to a dict."""
    raw = draft.get("metadata")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}


@mcp.tool()
def search_knowledge_base(query: str, collection: str = "campaigns", top_k: int = 5) -> str:
    """Semantic search over CraftAI's marketing knowledge base (ChromaDB).

    Args:
        query: what to look for (a topic, product, or message).
        collection: 'campaigns' (approved campaigns), 'social' (social posts),
            or 'guidelines' (brand guidelines).
        top_k: number of passages to return (default 5).
    Returns a JSON list of {relevance, text}.
    """
    name = _COLLECTIONS.get(collection)
    if not name:
        return f"Unknown collection '{collection}'. Choose one of: {', '.join(_COLLECTIONS)}."
    count = cc.collection_count(name)
    if count == 0:
        return f"The '{collection}' collection is empty."
    emb = embedder.encode_single(query)
    res = cc.query_collection(
        collection_name=name, query_embeddings=[emb], n_results=min(top_k, count)
    )
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    hits = [{"relevance": round(1.0 - d, 3), "text": doc} for doc, d in zip(docs, dists)]
    return json.dumps(hits, indent=2, ensure_ascii=False)


@mcp.tool()
def list_recent_briefs(limit: int = 10) -> str:
    """List recent campaign briefs (brief_id, brand, channel, status) so you can look one up."""
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "select brief_id, brand, channel, status from briefs order by rowid desc limit ?",
        (int(limit),),
    ).fetchall()
    conn.close()
    return json.dumps([dict(r) for r in rows], indent=2, ensure_ascii=False)


@mcp.tool()
def get_brief_status(brief_id: str) -> str:
    """Get a brief's status, latest draft, compliance score, and generated image URL."""
    brief = get_brief(brief_id)
    if not brief:
        return f"No brief found for id '{brief_id}'."
    draft = get_latest_draft(brief_id) or {}
    meta = _meta(draft)
    return json.dumps(
        {
            "brief_id": brief_id,
            "brand": brief.get("brand"),
            "channel": brief.get("channel"),
            "persona": brief.get("persona"),
            "status": brief.get("status"),
            "judge_score": draft.get("judge_score"),
            "compliance_pass": bool(draft.get("compliance_pass")),
            "image_url": meta.get("image_url"),
            "draft": (draft.get("content") or "")[:1500],
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool()
def get_campaign_image(brief_id: str) -> str:
    """Return the URL of the externally-stored (Cloudinary / local) campaign image for a brief."""
    draft = get_latest_draft(brief_id) or {}
    meta = _meta(draft)
    return meta.get("image_url") or f"No campaign image was generated for brief '{brief_id}'."


@mcp.tool()
def list_personas() -> str:
    """List the audience personas CraftAI tailors content to."""
    personas = get_personas()
    if personas:
        return json.dumps(
            [
                {
                    "name": p.get("name"),
                    "tone": p.get("preferred_tone"),
                    "age_range": p.get("age_range"),
                    "income": p.get("income_bracket"),
                }
                for p in personas
            ],
            indent=2,
            ensure_ascii=False,
        )
    return json.dumps(
        [
            "Premium Empty Nesters", "Smart-Saving Young Families", "Premium Professionals",
            "Value Families", "Connected Families", "Digital Professionals",
        ],
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()  # stdio transport (Claude Desktop and other MCP clients)
