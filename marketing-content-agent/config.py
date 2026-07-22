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

    # ── Judge LLM (optional; empty = reuse generator model) ──────────────────
    JUDGE_GROQ_MODEL: str = ""
    JUDGE_GROQ_API_KEY: str = ""

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── Storage ───────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    SQLITE_DB_PATH: str = "./data/craftai.db"

    # ── MLflow ────────────────────────────────────────────────────────────────
    # Use SQLite backend — the old file-store is deprecated in MLflow >= 2.16
    MLFLOW_TRACKING_URI: str = "sqlite:///./data/mlflow.db"

    # ── Agent thresholds ──────────────────────────────────────────────────────
    JUDGE_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)
    CRAG_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
    MAX_REVISIONS: int = Field(default=3, ge=1, le=10)
    # RAGAS is the heaviest token consumer (~4 LLM calls per evaluation). It now
    # runs ONCE on the final draft that reaches the human gate, not per revision.
    # Set false to skip it entirely (demos / tight API quota).
    RAGAS_ENABLED: bool = True

    # ── [image-based-campaign] Image feature (input vision + output generation) ─
    # Master feature flag — when False the pipeline behaves exactly as before:
    # the vision + art-director nodes become runtime no-ops (graph is unchanged).
    IMAGE_FEATURE_ENABLED: bool = False
    # Image OUTPUT (text → image). Cloudflare Workers AI by default (free tier);
    # switch IMAGE_PROVIDER to "gemini" for legible in-image text at demo quality.
    IMAGE_PROVIDER: str = "cloudflare"
    IMAGE_MODEL: str = "@cf/black-forest-labs/flux-1-schnell"
    CLOUDFLARE_ACCOUNT_ID: str = ""
    CLOUDFLARE_API_TOKEN: str = ""
    # Gemini image generation (Interactions API) — renders headline text properly.
    GEMINI_API_KEY: str = ""
    GEMINI_IMAGE_MODEL: str = "gemini-3.1-flash-image"
    IMAGE_ASPECT_RATIO: str = "1:1"
    IMAGE_SIZE: str = "1K"
    # Image INPUT (image → text). Groq vision model, reuses the Groq key.
    VISION_PROVIDER: str = "groq"
    VISION_MODEL: str = "qwen/qwen3.6-27b"   # only vision-capable model on Groq
    # Image COMPLIANCE JUDGE — a different vendor from IMAGE_PROVIDER so the model
    # that generated the image never grades its own output.
    # OFF by default: Groq's free-tier 8k TPM ceiling can't fit image + reasoning +
    # JSON answer reliably, so generated images are human-reviewed only.
    IMAGE_JUDGE_ENABLED: bool = False
    IMAGE_JUDGE_PROVIDER: str = "groq"
    IMAGE_JUDGE_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)
    # Image storage backend: "local" (disk, served via the API's /media mount) or
    # "cloudinary" (CDN-backed, permanent public URLs — needed for a deployed UI).
    IMAGE_STORAGE_BACKEND: str = "local"
    MEDIA_BASE_URL: str = "http://localhost:8000"
    # Cloudinary (only when IMAGE_STORAGE_BACKEND=cloudinary). Uploaded images get
    # a permanent, publicly-readable CDN URL — no signed-URL expiry to manage.
    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""
    CLOUDINARY_FOLDER: str = "craftai"

    # ── Server ports ──────────────────────────────────────────────────────────
    API_PORT: int = 8000
    UI_PORT: int = 8501

    # ── CORS ──────────────────────────────────────────────────────────────────
    ALLOWED_DOMAINS: str = "http://localhost:8501,http://127.0.0.1:8501"

    @model_validator(mode="after")
    def _ensure_data_dirs(self) -> Settings:
        os.makedirs(os.path.dirname(self.SQLITE_DB_PATH) or ".", exist_ok=True)
        os.makedirs(self.CHROMA_PERSIST_DIR, exist_ok=True)
        # MLflow URI is sqlite:///./data/mlflow.db — extract and create the data dir only
        mlflow_uri = self.MLFLOW_TRACKING_URI
        if mlflow_uri.startswith("sqlite:///"):
            db_path = mlflow_uri[len("sqlite:///"):]
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        elif not mlflow_uri.startswith(("http://", "https://")):
            os.makedirs(mlflow_uri, exist_ok=True)
        # [image-based-campaign] Ensure local image dirs exist (generated + uploads).
        data_root = os.path.dirname(self.SQLITE_DB_PATH) or "."
        os.makedirs(os.path.join(data_root, "images"), exist_ok=True)
        os.makedirs(os.path.join(data_root, "uploads"), exist_ok=True)
        return self

    @property
    def data_root(self) -> str:
        # [image-based-campaign] Root dir that holds images/ and uploads/.
        return os.path.dirname(self.SQLITE_DB_PATH) or "."

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
            max_tokens=4096,
        )

    if settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=temperature,
            num_predict=4096,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER!r}. Use 'groq' or 'ollama'.")


def get_judge_llm(temperature: float = 0.1):
    """LLM used for evaluation (LLM-as-judge + RAGAS).
    Falls back to the generator model when no judge model is configured.
    Currently Groq only."""
    judge_model = settings.JUDGE_GROQ_MODEL or settings.GROQ_MODEL
    judge_api_key = settings.JUDGE_GROQ_API_KEY or settings.GROQ_API_KEY

    if not judge_api_key:
        raise ValueError("Judge model needs a Groq API key (GROQ_API_KEY or JUDGE_GROQ_API_KEY).")

    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=judge_api_key,
        model=judge_model,
        temperature=temperature,
        max_tokens=4096,
    )


# [image-based-campaign] Vision LLM factory (image → text description / judging).
# NOTE: the Groq vision model is a *reasoning* model — it spends tokens on an
# internal <think> block before answering, so max_tokens must be generous or the
# reply is truncated before any real output. json_mode forces a parseable object.
def get_vision_llm(temperature: float = 0.2, max_tokens: int = 2048, json_mode: bool = False):
    """Return a vision-capable chat model. Groq only; reuses the Groq API key."""
    if settings.VISION_PROVIDER != "groq":
        raise ValueError(f"Unsupported VISION_PROVIDER: {settings.VISION_PROVIDER!r}.")
    if not settings.GROQ_API_KEY:
        raise ValueError("Vision model needs GROQ_API_KEY.")

    from langchain_groq import ChatGroq

    kwargs: dict = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.VISION_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs=kwargs,
    )
