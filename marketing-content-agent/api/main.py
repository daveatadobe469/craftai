from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from db.sqlite import create_tables
from rag.chroma_client import init_collections


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    app.state.tasks: dict[str, asyncio.Task] = {}
    app.state.queues: dict[str, asyncio.Queue] = {}

    create_tables()

    init_collections()

    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("craftai_campaigns")
    except Exception as exc:
        import warnings
        warnings.warn(f"MLflow setup failed (tracking disabled): {exc}")

    yield

    for task in app.state.tasks.values():
        if not task.done():
            task.cancel()
    pending = [t for t in app.state.tasks.values() if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


app = FastAPI(
    title="CRAFTAI — Intelligent Marketing Content Supply Chain",
    description=(
        "5-stage autonomous content pipeline: Brief → Draft → Comply → Approve → Index. "
        "Powered by LangGraph multi-agent orchestration, ChromaDB RAG, and MLflow observability."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routers import audit, brief, config, decision, ingest, search, status, stream

app.include_router(brief.router,    prefix="/api/v1", tags=["Brief"])
app.include_router(status.router,   prefix="/api/v1", tags=["Status"])
app.include_router(decision.router, prefix="/api/v1", tags=["Decision"])
app.include_router(search.router,   prefix="/api/v1", tags=["Search"])
app.include_router(stream.router,   prefix="/api/v1", tags=["Stream"])
app.include_router(ingest.router,   prefix="/api/v1", tags=["Ingest"])
app.include_router(config.router,   prefix="/api/v1", tags=["Config"])
app.include_router(audit.router,    prefix="/api/v1", tags=["Audit"])


@app.get("/health", tags=["Health"])
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "craftai-api"}
