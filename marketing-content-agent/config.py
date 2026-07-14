from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    LLM_PROVIDER: Literal["groq", "ollama"] = "groq"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama3-70b-8192"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── Storage ───────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    SQLITE_DB_PATH: str = "./data/craftai.db"

    # ── Auto-ingestion ───────────────────────────────────────────────────────
    # Files dropped here are embedded into the brand_guidelines collection
    # automatically on API startup — no manual upload needed.
    BRAND_GUIDELINE_FOLDER: str = "./brand_guideline"

    # ── MLflow ────────────────────────────────────────────────────────────────
    # Use SQLite backend — the old file-store is deprecated in MLflow >= 2.16
    MLFLOW_TRACKING_URI: str = "sqlite:///./data/mlflow.db"

    # ── Agent thresholds ──────────────────────────────────────────────────────
    JUDGE_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)
    CRAG_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
    MAX_REVISIONS: int = Field(default=3, ge=1, le=10)

    # ── Server ports ──────────────────────────────────────────────────────────
    API_PORT: int = 8000
    UI_PORT: int = 8501

    # ── CORS ──────────────────────────────────────────────────────────────────
    ALLOWED_DOMAINS: str = "http://localhost:8501,http://127.0.0.1:8501"

    @model_validator(mode="after")
    def _ensure_data_dirs(self) -> Settings:
        os.makedirs(os.path.dirname(self.SQLITE_DB_PATH) or ".", exist_ok=True)
        os.makedirs(self.CHROMA_PERSIST_DIR, exist_ok=True)
        os.makedirs(self.BRAND_GUIDELINE_FOLDER, exist_ok=True)
        # MLflow URI is sqlite:///./data/mlflow.db — extract and create the data dir only
        mlflow_uri = self.MLFLOW_TRACKING_URI
        if mlflow_uri.startswith("sqlite:///"):
            db_path = mlflow_uri[len("sqlite:///"):]
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        elif not mlflow_uri.startswith(("http://", "https://")):
            os.makedirs(mlflow_uri, exist_ok=True)
        return self

    @property
    def allowed_origins(self) -> list[str]:
        return [d.strip() for d in self.ALLOWED_DOMAINS.split(",") if d.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()


def get_llm(temperature: float = 0.7):
    """Factory that returns a configured LangChain chat model."""
    if settings.LLM_PROVIDER == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or set the environment variable."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=temperature,
        )

    if settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=temperature,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER!r}. Use 'groq' or 'ollama'.")
