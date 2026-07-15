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
    """Run migrations and seed cluster personas (P01–P06)."""
    sql = _MIGRATION.read_text(encoding="utf-8")
    with _connect() as conn:
        conn.executescript(sql)
    seed_cluster_personas()


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


def update_latest_draft_decision(
    brief_id: str,
    human_decision: str,
    human_edits: str | None,
    reviewed_by: str,
    reviewed_at: str,
) -> bool:
    """Persist human review on the latest draft row (before curator finishes)."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT draft_id FROM drafts
            WHERE brief_id=?
            ORDER BY revision_count DESC, created_at DESC
            LIMIT 1
            """,
            (brief_id,),
        ).fetchone()
        if row is None:
            return False
        conn.execute(
            """
            UPDATE drafts
            SET human_decision=?, human_edits=?, reviewed_by=?, reviewed_at=?
            WHERE draft_id=?
            """,
            (human_decision, human_edits, reviewed_by, reviewed_at, row["draft_id"]),
        )
    return True


def get_audit_event_data(brief_id: str, event_type: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT event_data FROM audit_log
            WHERE brief_id=? AND event_type=?
            ORDER BY id DESC LIMIT 1
            """,
            (brief_id, event_type),
        ).fetchone()
    if row is None:
        return None
    raw = row["event_data"]
    if isinstance(raw, str):
        return json.loads(raw or "{}")
    return dict(raw or {})


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

def _ensure_persona_columns(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(personas)")}
    if "persona_id" not in cols:
        conn.execute("ALTER TABLE personas ADD COLUMN persona_id TEXT")
    if "profile_json" not in cols:
        conn.execute(
            "ALTER TABLE personas ADD COLUMN profile_json TEXT NOT NULL DEFAULT '{}'"
        )


def _parse_persona_row(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["interests"] = json.loads(d.get("interests") or "[]")
    d["pain_points"] = json.loads(d.get("pain_points") or "[]")
    raw_profile = d.get("profile_json")
    if isinstance(raw_profile, str):
        profile = json.loads(raw_profile or "{}")
    elif isinstance(raw_profile, dict):
        profile = raw_profile
    else:
        profile = {}
    d["profile"] = profile
    if profile:
        cs = profile.get("content_strategy") or {}
        d["recommended_angles"] = cs.get("recommended_angles") or d["interests"]
        d["channel_playbook"] = cs.get("channel_playbook") or {}
        d["language_guardrails"] = cs.get("language_guardrails") or {}
        d["voice_attributes"] = cs.get("voice_attributes") or []
    return d


def upsert_persona(record: dict[str, Any]) -> None:
    profile = record.get("profile_json") or {}
    if not isinstance(profile, str):
        profile_json = json.dumps(profile)
    else:
        profile_json = profile
    sql = """
        INSERT INTO personas
            (name, description, age_range, income_bracket, interests, pain_points,
             preferred_tone, char_limit_email, char_limit_social, char_limit_linkedin,
             char_limit_ad, char_limit_blog, persona_id, profile_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            description=excluded.description,
            age_range=excluded.age_range,
            income_bracket=excluded.income_bracket,
            interests=excluded.interests,
            pain_points=excluded.pain_points,
            preferred_tone=excluded.preferred_tone,
            char_limit_email=excluded.char_limit_email,
            char_limit_social=excluded.char_limit_social,
            char_limit_linkedin=excluded.char_limit_linkedin,
            char_limit_ad=excluded.char_limit_ad,
            char_limit_blog=excluded.char_limit_blog,
            persona_id=excluded.persona_id,
            profile_json=excluded.profile_json
    """
    with _connect() as conn:
        conn.execute(sql, (
            record["name"],
            record["description"],
            record["age_range"],
            record["income_bracket"],
            json.dumps(record.get("interests") or []),
            json.dumps(record.get("pain_points") or []),
            record["preferred_tone"],
            record["char_limit_email"],
            record["char_limit_social"],
            record["char_limit_linkedin"],
            record["char_limit_ad"],
            record["char_limit_blog"],
            record.get("persona_id"),
            profile_json,
        ))


def seed_cluster_personas() -> int:
    """Load P01–P06 cluster personas from data/personas into SQLite."""
    from db.persona_seed import load_persona_files, load_personas_all

    with _connect() as conn:
        _ensure_persona_columns(conn)

    records = load_persona_files()
    if len(records) < 6:
        records = load_personas_all()
    with _connect() as conn:
        # Replace cluster personas on rename (upsert is keyed by name, not persona_id)
        conn.execute(
            "DELETE FROM personas WHERE persona_id IN ('P01','P02','P03','P04','P05','P06')"
        )
    for record in records:
        upsert_persona(record)
    # Remove legacy demo personas (pre-cluster seed names without persona_id)
    with _connect() as conn:
        conn.execute("DELETE FROM personas WHERE persona_id IS NULL OR persona_id = ''")
    return len(records)


def get_personas() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM personas ORDER BY persona_id, name"
        ).fetchall()
    return [_parse_persona_row(row) for row in rows]


def get_persona(name: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM personas WHERE name=?", (name,)).fetchone()
    if row is None:
        return None
    return _parse_persona_row(row)
