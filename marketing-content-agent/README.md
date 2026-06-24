# Marketing Content Agent

A multi-agent system for generating, reviewing, and curating marketing content
using **LangGraph**, **FastAPI**, **Streamlit**, and **ChromaDB**.

## Quick Start

```bash
# 1. Clone and enter the project
cd marketing-content-agent

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 3. Configure environment
cp .env.example .env           # add your GROQ_API_KEY

# 4. Install dependencies
pip install -r requirements.txt
```

## API Reference (to be updated)

| Method | Path                    | Description                  |
|--------|-------------------------|------------------------------|
| POST   | `/api/v1/brief`         | Submit a new content brief   |
| GET    | `/api/v1/status/{id}`   | Poll run status              |
| POST   | `/api/v1/decision/{id}` | Submit human review decision |
| POST   | `/api/v1/search`        | Semantic content search      |
| GET    | `/api/v1/stream/{id}`   | SSE live trace               |

## Project Structure

```
marketing-content-agent/
├── graph/                     # LangGraph agent graph
│   ├── state.py               # AgentState TypedDict
│   ├── builder.py             # StateGraph assembly + compile()
│   └── nodes/
│       ├── orchestrator.py    # Agent A
│       ├── generator.py       # Agent B — RAG + generation
│       ├── compliance.py      # Agent C — rule tools + judge
│       ├── human_gate.py      # Human interrupt logic
│       └── curator.py         # Agent D — index + audit
├── api/                       # FastAPI application
│   ├── main.py                # app = FastAPI(); mount routers
│   ├── routers/
│   │   ├── brief.py           # POST /api/v1/brief
│   │   ├── status.py          # GET  /api/v1/status/{id}
│   │   ├── decision.py        # POST /api/v1/decision/{id}
│   │   ├── search.py          # POST /api/v1/search
│   │   └── stream.py          # GET  /api/v1/stream/{id} SSE
│   └── schemas/
│       ├── brief.py           # BriefPayload, BriefResponse
│       └── decision.py        # DecisionPayload
├── ui/                        # Streamlit application
│   ├── app.py
│   ├── pages/
│   │   ├── creator.py
│   │   ├── reviewer.py
│   │   └── search.py
│   └── components/trace.py
├── rag/                       # RAG utilities
│   ├── chroma_client.py       # Singleton ChromaDB client
│   ├── retriever.py           # HyDE + CRAG retrieval logic
│   ├── embedder.py            # SentenceTransformer wrapper
│   └── evaluator.py           # RAGAS metric computation
├── compliance/
│   ├── tools.py               # check_restricted_words() etc.
│   └── judge.py               # LLM-as-judge prompt + scorer
├── db/
│   ├── sqlite.py              # SQLite CRUD functions
│   └── migrations/            # SQL schema files
├── prompts/                   # Jinja2 prompt templates
│   ├── email.j2  linkedin.j2  social.j2  ad.j2  blog.j2
├── data/
│   ├── raw/                   # Kaggle CSVs, social ad JSONs
│   ├── synthetic/             # Generated brand guideline PDFs
│   └── chroma/                # ChromaDB persistent storage
├── scripts/
│   ├── init_chroma.py         # Create + populate all 3 collections
│   ├── generate_guidelines.py # Faker → Markdown → PDF pipeline
│   ├── cluster_personas.py    # K-Means → personas table
│   └── reindex_chroma.py      # Full re-embed after model switch
├── tests/
│   ├── test_compliance.py
│   ├── test_ragas.py
│   └── test_graph.py
├── .env.example
├── requirements.txt
├── Makefile                   # make dev / make test / make index
└── README.md

```



## Environment Variables

| Variable            | Default            | Description                        |
|---------------------|--------------------|------------------------------------|
| `GROQ_API_KEY`      | —                  | Required — your GROQ API key       |
| `API_BASE_URL`      | `http://localhost:8000` | FastAPI base URL              |
| `SQLITE_PATH`       | `data/agent.db`    | SQLite database path               |
| `CHROMA_DIR`        | `data/chroma`      | ChromaDB persistence directory     |
| `GUIDELINE_COUNT`   | `5`                | Number of synthetic guidelines     |
| `PERSONA_CLUSTERS`  | `5`                | K for persona clustering           |

