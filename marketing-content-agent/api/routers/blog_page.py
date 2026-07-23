from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from db.sqlite import get_brief, get_latest_draft
from services.blog_html_renderer import render_blog_html

router = APIRouter()


@router.get("/blog-page/{brief_id}", response_class=HTMLResponse)
async def get_blog_page(brief_id: str) -> HTMLResponse:
    """
    Render the approved/latest draft of a 'blog' channel brief as a
    self-contained, responsive HTML page (embedded CSS) ready to view
    or download.
    """
    brief = get_brief(brief_id)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"Brief '{brief_id}' not found.")

    if brief.get("channel", "").lower() != "blog":
        raise HTTPException(
            status_code=400,
            detail=f"Brief '{brief_id}' is channel '{brief.get('channel')}', not 'blog'.",
        )

    draft_row = get_latest_draft(brief_id)
    if draft_row is None:
        raise HTTPException(status_code=404, detail=f"No draft found for brief '{brief_id}'.")

    draft_content = draft_row.get("content", "")
    human_edits = draft_row.get("human_edits")
    if human_edits and str(human_edits).strip():
        draft_content = str(human_edits).strip()

    html = render_blog_html(
        brand=brief.get("brand", ""),
        persona=brief.get("persona", ""),
        key_message=brief.get("key_message", ""),
        draft_metadata=draft_row.get("metadata") or {},
        fallback_draft_text=draft_content,
    )
    return HTMLResponse(content=html)
