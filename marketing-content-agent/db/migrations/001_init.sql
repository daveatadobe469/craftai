-- ─── CRAFTAI Database Schema ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS briefs (
    brief_id    TEXT PRIMARY KEY,
    brand       TEXT NOT NULL,
    channel     TEXT NOT NULL CHECK(channel IN ('email','linkedin','social','ad','blog')),
    persona     TEXT NOT NULL,
    key_message TEXT NOT NULL,
    constraints TEXT NOT NULL DEFAULT '{}',   -- JSON blob
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS drafts (
    draft_id        TEXT PRIMARY KEY,
    brief_id        TEXT NOT NULL REFERENCES briefs(brief_id),
    revision_count  INTEGER NOT NULL DEFAULT 0,
    content         TEXT NOT NULL,
    metadata        TEXT NOT NULL DEFAULT '{}',   -- JSON blob
    judge_score     REAL,
    compliance_pass INTEGER NOT NULL DEFAULT 0,   -- 0 = false, 1 = true
    human_decision  TEXT CHECK(human_decision IN ('approved','edited','rejected') OR human_decision IS NULL),
    human_edits     TEXT,
    reviewed_by     TEXT,
    reviewed_at     TEXT,
    ragas_scores    TEXT NOT NULL DEFAULT '{}',   -- JSON blob
    mlflow_run_id   TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    brief_id        TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    event_data      TEXT NOT NULL DEFAULT '{}',   -- JSON blob
    actor           TEXT NOT NULL DEFAULT 'system',
    ts              TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS personas (
    name            TEXT PRIMARY KEY,
    description     TEXT NOT NULL,
    age_range       TEXT NOT NULL DEFAULT '25-45',
    income_bracket  TEXT NOT NULL DEFAULT 'middle',
    interests       TEXT NOT NULL DEFAULT '[]',   -- JSON array
    pain_points     TEXT NOT NULL DEFAULT '[]',   -- JSON array
    preferred_tone  TEXT NOT NULL DEFAULT 'professional',
    char_limit_email    INTEGER NOT NULL DEFAULT 500,
    char_limit_social   INTEGER NOT NULL DEFAULT 280,
    char_limit_linkedin INTEGER NOT NULL DEFAULT 700,
    char_limit_ad       INTEGER NOT NULL DEFAULT 150,
    char_limit_blog     INTEGER NOT NULL DEFAULT 2000
);

-- ─── Seed Personas ────────────────────────────────────────────────────────────

INSERT OR IGNORE INTO personas
    (name, description, age_range, income_bracket, interests, pain_points, preferred_tone,
     char_limit_email, char_limit_social, char_limit_linkedin, char_limit_ad, char_limit_blog)
VALUES
    (
        'Budget-Conscious',
        'Price-sensitive shoppers who prioritise value for money and deals.',
        '25-40',
        'lower-middle',
        '["discounts","cashback","bulk buying","coupons"]',
        '["unexpected expenses","price hikes","hidden fees"]',
        'friendly',
        400, 240, 600, 120, 1500
    ),
    (
        'Premium Buyer',
        'High-income consumers who value quality, exclusivity, and brand prestige.',
        '35-55',
        'upper',
        '["luxury goods","travel","fine dining","exclusive experiences"]',
        '["poor quality","lack of exclusivity","generic marketing"]',
        'sophisticated',
        600, 280, 800, 160, 2500
    ),
    (
        'Family Planner',
        'Parents and caregivers focused on family well-being, safety, and long-term planning.',
        '28-45',
        'middle',
        '["family activities","education","health","home improvement"]',
        '["time constraints","budget management","child safety"]',
        'warm',
        500, 260, 700, 140, 2000
    ),
    (
        'Young Professional',
        'Career-driven millennials and Gen-Z who value innovation, convenience, and social impact.',
        '22-32',
        'middle',
        '["technology","sustainability","career growth","social media"]',
        '["work-life balance","student debt","housing costs"]',
        'energetic',
        450, 280, 650, 130, 1800
    );

-- ─── Indexes ─────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_drafts_brief_id ON drafts(brief_id);
CREATE INDEX IF NOT EXISTS idx_audit_brief_id  ON audit_log(brief_id);
CREATE INDEX IF NOT EXISTS idx_briefs_status   ON briefs(status);
