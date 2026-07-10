from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from config import settings

_DB_PATH = settings.SQLITE_DB_PATH
_MIGRATION = Path(__file__).parent / "migrations" / "001_init.sql"


def _connect() -> sqlite3.Connection:
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def create_tables() -> None:
    """Run the migration DDL if tables do not yet exist."""
    sql = _MIGRATION.read_text(encoding="utf-8")
    with _connect() as conn:
        conn.executescript(sql)


# ─── Briefs ───────────────────────────────────────────────────────────────────

def write_brief(
    brief_id: str,
    brand: str,
    channel: str,
    persona: str,
    key_message: str,
    constraints: dict[str, Any] | None = None,
) -> None:
    sql = """
        INSERT INTO briefs (brief_id, brand, channel, persona, key_message, constraints, status)
        VALUES (?, ?, ?, ?, ?, ?, 'pending')
        ON CONFLICT(brief_id) DO UPDATE SET
            brand=excluded.brand, channel=excluded.channel, persona=excluded.persona,
            key_message=excluded.key_message, constraints=excluded.constraints,
            updated_at=datetime('now')
    """
    with _connect() as conn:
        conn.execute(sql, (
            brief_id, brand, channel, persona, key_message,
            json.dumps(constraints or {}),
        ))


def update_brief_status(brief_id: str, status: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE briefs SET status=?, updated_at=datetime('now') WHERE brief_id=?",
            (status, brief_id),
        )


def get_brief(brief_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM briefs WHERE brief_id=?", (brief_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["constraints"] = json.loads(d.get("constraints") or "{}")
    return d


# ─── Drafts ───────────────────────────────────────────────────────────────────

def write_draft(
    brief_id: str,
    content: str,
    revision_count: int = 0,
    metadata: dict[str, Any] | None = None,
    judge_score: float | None = None,
    compliance_pass: bool = False,
    human_decision: str | None = None,
    human_edits: str | None = None,
    reviewed_by: str | None = None,
    reviewed_at: str | None = None,
    ragas_scores: dict[str, Any] | None = None,
    mlflow_run_id: str | None = None,
) -> str:
    draft_id = str(uuid.uuid4())
    sql = """
        INSERT INTO drafts
            (draft_id, brief_id, revision_count, content, metadata, judge_score,
             compliance_pass, human_decision, human_edits, reviewed_by, reviewed_at,
             ragas_scores, mlflow_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _connect() as conn:
        conn.execute(sql, (
            draft_id, brief_id, revision_count, content,
            json.dumps(metadata or {}),
            judge_score,
            1 if compliance_pass else 0,
            human_decision, human_edits, reviewed_by, reviewed_at,
            json.dumps(ragas_scores or {}),
            mlflow_run_id,
        ))
    return draft_id


def get_latest_draft(brief_id: str) -> dict[str, Any] | None:
    sql = """
        SELECT * FROM drafts WHERE brief_id=?
        ORDER BY revision_count DESC, created_at DESC LIMIT 1
    """
    with _connect() as conn:
        row = conn.execute(sql, (brief_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["metadata"] = json.loads(d.get("metadata") or "{}")
    d["ragas_scores"] = json.loads(d.get("ragas_scores") or "{}")
    d["compliance_pass"] = bool(d.get("compliance_pass"))
    return d


# ─── Audit Log ────────────────────────────────────────────────────────────────

def write_audit(
    brief_id: str,
    event_type: str,
    event_data: dict[str, Any] | None = None,
    actor: str = "system",
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO audit_log (brief_id, event_type, event_data, actor) VALUES (?, ?, ?, ?)",
            (brief_id, event_type, json.dumps(event_data or {}), actor),
        )


# ─── Personas ─────────────────────────────────────────────────────────────────

def get_personas() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM personas").fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["interests"] = json.loads(d.get("interests") or "[]")
        d["pain_points"] = json.loads(d.get("pain_points") or "[]")
        result.append(d)
    return result


def upsert_persona(
    name: str,
    description: str,
    age_range: str = "25-45",
    income_bracket: str = "middle",
    interests: list[str] | None = None,
    pain_points: list[str] | None = None,
    preferred_tone: str = "professional",
    char_limit_email: int = 500,
    char_limit_social: int = 280,
    char_limit_linkedin: int = 700,
    char_limit_ad: int = 150,
    char_limit_blog: int = 2000,
) -> None:
    """Insert or update a persona by name (mirrors write_brief's ON CONFLICT pattern)."""
    sql = """
        INSERT INTO personas
            (name, description, age_range, income_bracket, interests, pain_points,
             preferred_tone, char_limit_email, char_limit_social, char_limit_linkedin,
             char_limit_ad, char_limit_blog)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            description=excluded.description, age_range=excluded.age_range,
            income_bracket=excluded.income_bracket, interests=excluded.interests,
            pain_points=excluded.pain_points, preferred_tone=excluded.preferred_tone,
            char_limit_email=excluded.char_limit_email,
            char_limit_social=excluded.char_limit_social,
            char_limit_linkedin=excluded.char_limit_linkedin,
            char_limit_ad=excluded.char_limit_ad,
            char_limit_blog=excluded.char_limit_blog
    """
    with _connect() as conn:
        conn.execute(sql, (
            name, description, age_range, income_bracket,
            json.dumps(interests or []),
            json.dumps(pain_points or []),
            preferred_tone,
            char_limit_email, char_limit_social, char_limit_linkedin,
            char_limit_ad, char_limit_blog,
        ))


def get_persona(name: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM personas WHERE name=?", (name,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["interests"] = json.loads(d.get("interests") or "[]")
    d["pain_points"] = json.loads(d.get("pain_points") or "[]")
    return d
