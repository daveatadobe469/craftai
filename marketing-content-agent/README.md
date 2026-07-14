<div align="center">
  <img src="LOGO/Craft AI Logo.png" alt="CraftAI Logo" width="120"/>

  # CraftAI — Intelligent Marketing Content Supply Chain

  **Content · Review · Authoring · Flow-through AI**

  A multi-agent system that autonomously **generates, reviews, human-approves, and curates**
  brand-compliant marketing content — powered by **LangGraph**, **Agentic RAG (ChromaDB)**,
  **FastAPI**, **Streamlit**, and **MLflow**.
</div>

---

## Project Team
|SNo | Name |
|-----|-------|
|1| Abhinao Shrivastava |
|2| Amit Pandey |
|3| Asha Pathik |
|4| Devendra Dave |
|5| Dipen Sen |
|6| Guruprasad Ramamoorthi |
|7| Harish P |
|8| Jeevan Eranti |
|9| Narayana Murthy |
|10| Piyush Sinha |
|11| Sunil Kumar Sahu |

---


## 1. Project Purpose

CraftAI models the full marketing-content lifecycle as a **5-stage autonomous supply chain**:

> **Brief → Draft → Comply → Approve → Index**

A marketer submits a short *campaign brief* (brand, channel, persona, key message). CraftAI then:

1. **Orchestrates** the run — validates the brief, loads the target persona, opens an MLflow run.
2. **Generates** a channel-specific draft using **HyDE + CRAG Agentic RAG** over a knowledge base of
   approved campaigns, social posts, and brand guidelines, then scores it with **RAGAS**.
3. **Checks compliance** in two passes — deterministic rule tools (restricted words, char limits,
   URL safety) *plus* an **LLM-as-judge** for holistic brand alignment. Non-compliant drafts loop
   back to the generator (up to `MAX_REVISIONS`).
4. **Pauses at a Human Gate** — a reviewer approves, edits, or rejects the draft (30-minute timeout).
5. **Curates** approved content — chunks, embeds, and upserts it back into the vector store so future
   generations learn from it, closing the feedback loop.

Every stage streams live progress over **Server-Sent Events (SSE)**, is persisted to **SQLite**, and
is logged to **MLflow** for full observability and auditability.

### Key Features
- 🤖 **Multi-agent LangGraph pipeline** with conditional revision + human-in-the-loop edges.
- 🔍 **Agentic RAG** — HyDE query rewriting + CRAG self-grading over three ChromaDB collections.
- 🛡️ **Dual-layer compliance** — deterministic rules + few-shot LLM-as-judge scoring.
- 📊 **RAGAS evaluation** (faithfulness, answer relevancy, context precision/recall) per draft.
- 📈 **MLflow tracking** of params, scores, and latencies.
- ⚙️ **Pluggable LLM** — Groq (cloud) or Ollama (local), switchable from the UI at runtime.
- 🖥️ **Rich Streamlit console** — LLM config, document ingestion, live pipeline tracker, review gate, audit dashboard.
- 📥 **Document ingestion** — upload PDF/DOCX/TXT/MD/CSV/JSON to grow the knowledge base.

---

## 2. Architecture at a Glance

```
                       ┌─────────────────────────────────────────────┐
                       │                Streamlit UI                  │
                       │  Config · Upload · Content Studio · Audit    │
                       └───────────────────────┬─────────────────────┘
                                                │ HTTP / SSE
                       ┌────────────────────────▼─────────────────────┐
                       │                 FastAPI  (api/)              │
                       │  /brief /status /decision /search /stream    │
                       │  /ingest /config /audit  /health             │
                       └───────────────────────┬──────────────────────┘
                                                │ astream()
   START → orchestrator → generator → compliance ─(revise)→ generator
                                          │
                                       (gate)
                                          ▼
                                     human_gate ─(approved)→ curator → END
                                          └────────(rejected)────────→ END

   Supporting layers:  graph/ (agents) · rag/ (HyDE+CRAG+RAGAS) ·
                       compliance/ (rules+judge) · db/ (SQLite) ·
                       ChromaDB (vectors) · MLflow (tracking)
```

---

## 3. Tech Stack & Tools Required

| Category            | Tool / Library                                              |
|---------------------|-------------------------------------------------------------|
| **Language**        | Python **3.10+** (3.11 recommended)                         |
| **Agent framework** | LangGraph, LangChain (core / community / text-splitters)    |
| **LLM providers**   | Groq (`langchain-groq`) *or* Ollama (`langchain-ollama`)    |
| **API**             | FastAPI, Uvicorn, SSE-Starlette, HTTPX, python-multipart    |
| **UI**              | Streamlit                                                   |
| **Vector DB**       | ChromaDB (persistent) + SentenceTransformers embeddings     |
| **RAG eval**        | RAGAS, HuggingFace Datasets                                 |
| **Observability**   | MLflow (SQLite backend)                                     |
| **Persistence**     | SQLite (WAL mode)                                           |
| **Templating**      | Jinja2 (per-channel prompt templates)                       |
| **Data science**    | scikit-learn, pandas, numpy, Faker                          |
| **Doc parsing**     | pypdf, pdfminer.six, python-docx                            |
| **Testing**         | pytest, pytest-asyncio, pytest-mock                         |
| **Tooling**         | ruff (lint), mypy (types), Make                             |

**External requirements:**
- A **Groq API key** (free at [console.groq.com](https://console.groq.com)) — *or* a running
  **Ollama** server (`ollama serve`) with a pulled model for fully local operation.
- Internet access on first run to download the SentenceTransformer embedding model.

---

## 4. Setup Instructions

### 4.1 Prerequisites
- Python 3.10+ and `pip`
- (Optional) `make` for the convenience targets
- A Groq API key **or** a local Ollama install

### 4.2 Install

```bash
# 1. Enter the project
cd marketing-content-agent

# 2. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows (PowerShell/CMD)
source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env       # Windows
cp .env.example .env         # Linux / macOS
#   → open .env and set GROQ_API_KEY=gsk_...
```

### 4.3 Seed the Knowledge Base (first run)

```bash
python scripts/init_chroma.py          # create + seed campaigns & social collections
python scripts/generate_guidelines.py  # seed brand-guideline documents
#   …or simply:  make index
```

> `python main.py` auto-seeds on first launch if `data/craftai.db` is empty.

### 4.4 Run

```bash
# Recommended — launches API + UI together with a colored console
python main.py
python main.py --api-only      # backend only
python main.py --skip-seed     # skip the seeding check

# …or via Make
make dev      # API + UI
make api      # FastAPI only
make ui       # Streamlit only
```

| Service | URL                              |
|---------|----------------------------------|
| UI      | http://localhost:8501            |
| API     | http://localhost:8000            |
| Docs    | http://localhost:8000/docs       |

### 4.5 Test & Lint

```bash
make test              # all tests           (pytest tests/ -v)
make test-compliance   # compliance unit tests
make test-ragas        # RAGAS tests (needs GROQ_API_KEY)
make test-graph        # graph integration tests
make lint              # ruff --fix + mypy
make clean             # remove generated data/ artifacts
```

---

## 5. API Reference

Base path: **`/api/v1`** (except `/health`) — full interactive docs at `/docs`, ReDoc at `/redoc`.

| Method | Path                              | Description                                        |
|--------|-----------------------------------|----------------------------------------------------|
| GET    | `/health`                         | Liveness probe (returns service status)            |
| POST   | `/api/v1/brief`                   | Submit a campaign brief; starts the pipeline (202) |
| GET    | `/api/v1/status/{brief_id}`       | Latest draft + compliance / RAGAS status           |
| POST   | `/api/v1/decision/{brief_id}`     | Submit human decision (approved / edited / rejected) |
| GET    | `/api/v1/stream/{brief_id}`       | SSE live agent trace (with heartbeats)             |
| POST   | `/api/v1/search`                  | Semantic search over a ChromaDB collection         |
| POST   | `/api/v1/ingest`                  | Upload + embed a document into a collection (201)  |
| GET    | `/api/v1/ingest/{collection}`     | Document count + sample for a collection           |
| DELETE | `/api/v1/ingest/{collection}/{doc_id}` | Delete all chunks of a document               |
| GET    | `/api/v1/config`                  | Read current LLM + embedding + threshold config    |
| POST   | `/api/v1/config`                  | Persist config to `.env` and hot-reload settings   |
| GET    | `/api/v1/config/ollama/models`    | List installed Ollama models                       |
| GET    | `/api/v1/config/groq/models`      | List supported Groq models                         |
| GET    | `/api/v1/config/embedding/models` | List recommended embedding models                  |
| POST   | `/api/v1/config/test`             | Test an LLM connection without saving              |
| GET    | `/api/v1/audit/events`            | Paginated audit log (filter by brief/type/actor)   |
| GET    | `/api/v1/audit/briefs`            | List submitted briefs (filter by status/brand/channel) |
| GET    | `/api/v1/audit/drafts`            | List draft records (filter by brief/decision)      |
| GET    | `/api/v1/audit/stats`             | Aggregate pipeline statistics / KPIs               |
| GET    | `/api/v1/audit/event-types`       | Distinct event types + actors (for filters)        |

---

## 6. Environment Variables

Read via `config.py` (Pydantic `BaseSettings`). `.env` keys are **case-insensitive**; unknown keys are ignored.

| Variable               | Default                        | Description                              |
|------------------------|--------------------------------|------------------------------------------|
| `LLM_PROVIDER`         | `groq`                         | LLM backend: `groq` or `ollama`          |
| `GROQ_API_KEY`         | `""` (empty)                   | Required for the Groq provider           |
| `GROQ_MODEL`           | `llama3-70b-8192`              | Groq model id                            |
| `OLLAMA_BASE_URL`      | `http://localhost:11434`       | Local Ollama endpoint                    |
| `OLLAMA_MODEL`         | `llama3`                       | Ollama model name                        |
| `EMBEDDING_MODEL`      | `all-MiniLM-L6-v2`             | SentenceTransformer embedding model      |
| `CHROMA_PERSIST_DIR`   | `./data/chroma`                | ChromaDB persistence directory           |
| `SQLITE_DB_PATH`       | `./data/craftai.db`            | SQLite database path                     |
| `MLFLOW_TRACKING_URI`  | `sqlite:///./data/mlflow.db`   | MLflow tracking backend (SQLite)         |
| `JUDGE_THRESHOLD`      | `0.7`                          | Min LLM-judge score to pass compliance (0–1) |
| `CRAG_THRESHOLD`       | `0.5`                          | Min CRAG relevance score to keep a RAG chunk (0–1) |
| `MAX_REVISIONS`        | `3`                            | Generate↔Comply loops before escalating (1–10) |
| `API_PORT`             | `8000`                         | FastAPI port                             |
| `UI_PORT`              | `8501`                         | Streamlit port                           |
| `ALLOWED_DOMAINS`      | `http://localhost:8501,http://127.0.0.1:8501` | Comma-separated CORS allow-list |

> **Note:** `config.py` auto-creates the `data/` directories for SQLite, ChromaDB, and MLflow on startup.
> The `.env.example` shipped with the repo lists several extra placeholder keys (OpenAI, Gemini, Anthropic,
> Pinecone, HuggingFace, SMTP email) that are **not** consumed by the core pipeline — only `GROQ_API_KEY`
> (or a running Ollama) is required.

---

## 7. Project Structure

```
marketing-content-agent/
│
├── main.py                     # App launcher — starts FastAPI + Streamlit, auto-seeds KB, streams logs
├── config.py                   # Pydantic Settings + get_llm() factory (Groq/Ollama switch)
├── requirements.txt            # Python dependencies
├── Makefile                    # dev / api / ui / index / cluster / test / lint / clean / init targets
├── Dockerfile                  # Container build (placeholder / to be completed)
├── pytest.ini                  # Pytest config (asyncio auto mode)
├── .env.example                # Template environment file — copy to .env
├── .gitignore                  # Ignores venv, data/, .env, caches
├── README.md                   # This file
├── LOGO/
│   └── Craft AI Logo.png        # App logo shown in the Streamlit header
│
├── graph/                      # ── LangGraph multi-agent state machine ──
│   ├── state.py                #   AgentState TypedDict — the shared pipeline state
│   ├── builder.py              #   Assembles nodes + conditional edges; compiles the graph
│   └── nodes/
│       ├── orchestrator.py     #   Stage 1 — validate brief, load persona, open MLflow run, build plan
│       ├── generator.py        #   Stage 2 — HyDE+CRAG retrieval, Jinja2 prompt, LLM draft, RAGAS eval
│       ├── compliance.py       #   Stage 3 — rule tools + LLM-judge; routes revise↔gate; persists draft
│       ├── human_gate.py       #   Stage 4 — suspends graph awaiting human decision (30-min timeout)
│       └── curator.py          #   Stage 5 — chunk, embed & upsert approved content; audit + MLflow
│
├── api/                        # ── FastAPI backend ──
│   ├── main.py                 #   App factory, lifespan (DB/Chroma/MLflow init), CORS, router mounts
│   ├── routers/
│   │   ├── brief.py            #   POST /brief — persist + kick off async graph run, wire SSE queue
│   │   ├── status.py           #   GET  /status/{id} — latest draft + compliance/RAGAS snapshot
│   │   ├── decision.py         #   POST /decision/{id} — unblock the human gate
│   │   ├── search.py           #   POST /search — semantic search over a collection
│   │   ├── stream.py           #   GET  /stream/{id} — SSE trace with heartbeats
│   │   ├── ingest.py           #   Upload/list/delete documents (PDF/DOCX/TXT/MD/CSV/JSON)
│   │   ├── config.py           #   Read/write LLM config to .env, list models, test connection
│   │   └── audit.py            #   Audit events, briefs, drafts, and aggregate pipeline stats
│   └── schemas/
│       ├── brief.py            #   BriefPayload / BriefResponse (Pydantic)
│       └── decision.py         #   DecisionPayload, SearchRequest/Result/Response
│
├── rag/                        # ── Retrieval-Augmented Generation layer ──
│   ├── chroma_client.py        #   Singleton ChromaDB client + collection helpers (3 collections)
│   ├── embedder.py             #   Cached SentenceTransformer encode()/encode_single()
│   ├── retriever.py            #   HyDE rewrite + CRAG self-grading retrieval pipeline
│   └── evaluator.py            #   RAGAS metric computation (faithfulness, relevancy, precision/recall)
│
├── compliance/                 # ── Brand-compliance engine ──
│   ├── tools.py                #   Deterministic checks: restricted words, char limits, URLs, required phrases
│   └── judge.py                #   Few-shot LLM-as-judge scorer → (score, evidence)
│
├── prompts/                    # ── Jinja2 prompt templates (one per channel) ──
│   ├── email.j2                #   Email draft prompt (JSON output: subject/body/cta)
│   ├── linkedin.j2             #   LinkedIn post prompt
│   ├── social.j2               #   Social media post prompt
│   ├── ad.j2                   #   Digital ad prompt
│   └── blog.j2                 #   Blog post prompt
│
├── db/                         # ── SQLite persistence ──
│   ├── sqlite.py               #   CRUD for briefs, drafts, audit log, personas
│   └── migrations/
│       └── 001_init.sql        #   Schema DDL + seed personas + indexes
│
├── scripts/                    # ── One-off seeding / maintenance scripts ──
│   ├── init_chroma.py          #   Create collections + seed approved_campaigns & social_content
│   ├── generate_guidelines.py  #   Seed brand_guidelines collection (synthetic content)
│   ├── cluster_personas.py     #   K-Means + PCA over campaign embeddings → cluster report JSON
│   └── reindex_chroma.py       #   Full re-embed after an embedding-model switch (placeholder)
│
├── ui/                         # ── Streamlit frontend ──
│   ├── app.py                  #   Main app: CSS, header, tabs (Config, Upload, Content Studio, Audit)
│   ├── app.ppy                 #   Backup/scratch copy of app.py (not executed)
│   ├── components/
│   │   └── trace.py            #   (placeholder) shared trace component
│   └── pages/
│       ├── creator.py          #   Brief form + 7-step live SSE pipeline tracker + inline review gate
│       ├── reviewer.py         #   Standalone draft review + approve/edit/reject by Brief ID
│       ├── audit.py            #   Audit dashboard — KPIs, briefs, drafts, events
│       ├── upload.py           #   Document ingestion page (collection targets + help)
│       ├── search.py           #   Knowledge-base semantic search page
│       └── settings.py         #   LLM/model settings page
│
├── tests/                      # ── Test suite ──
│   ├── conftest.py             #   Test env setup (isolated test DB / Chroma / MLflow paths)
│   ├── test_compliance.py      #   Unit tests for deterministic compliance rules
│   ├── test_graph.py           #   Integration tests for the LangGraph nodes/pipeline
│   └── test_ragas.py           #   RAGAS evaluation tests
│
└── data/                       # ── Generated at runtime (git-ignored) ──
    ├── chroma/                 #   ChromaDB persistent vector store
    ├── craftai.db              #   SQLite app database (briefs/drafts/audit/personas)
    └── mlflow.db               #   MLflow SQLite tracking backend
```

---

## 8. How the Pipeline Flows (End-to-End)

1. **UI → `POST /api/v1/brief`** persists the brief and launches `compiled_graph.astream(state)` as an async task, wiring an SSE queue.
2. **orchestrator** validates fields, loads the persona from SQLite (falls back to a default), starts an MLflow run.
3. **generator** runs HyDE+CRAG retrieval, renders the channel's Jinja2 prompt, calls the LLM, parses JSON, and scores with RAGAS.
4. **compliance** runs rule tools + LLM-judge; if it fails and revisions remain, it routes back to **generator**; otherwise to **human_gate**. The draft is persisted so the UI can display it.
5. **human_gate** blocks on an `asyncio.Event` until `POST /decision/{id}` (or a 30-min timeout → auto-reject).
6. On **approve/edit → curator** chunks, embeds, and upserts the final content into `approved_campaigns`; on **reject → END**.
7. The UI's live tracker consumes the SSE stream (`trace`, `human_action_required`, `pipeline_complete`, `done`, `error`) to advance the 7-step visual.

---

## 9. Personas (Seeded)

`db/migrations/001_init.sql` seeds four personas, each with tone + per-channel character limits:
**Budget-Conscious**, **Premium Buyer**, **Family Planner**, **Young Professional**.

---

## 10. Notes & Caveats
- `.env.example` currently contains generic placeholder keys — the app only requires **`GROQ_API_KEY`** (or a running Ollama). Other keys (OpenAI/Gemini/Anthropic/Pinecone/HF/email) are not used by the core pipeline.
- `Dockerfile`, `scripts/reindex_chroma.py`, and `ui/components/trace.py` are placeholders.
- `ui/app.ppy` is a non-executed backup of `app.py`.
- Switching the embedding model requires re-indexing the knowledge base (`make index`).

----