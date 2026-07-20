from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_ENV_PATH = Path(__file__).parent.parent.parent / ".env"

# ── Pydantic models ───────────────────────────────────────────────────────────

class LLMConfig(BaseModel):
    provider: Literal["groq", "ollama"]
    groq_api_key: Optional[str] = None
    groq_model: str = "llama3-70b-8192"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    embedding_model: str = "all-MiniLM-L6-v2"
    judge_threshold: float = 0.7
    crag_threshold: float = 0.5
    max_revisions: int = 3


class LLMConfigResponse(BaseModel):
    provider: str
    groq_model: str
    groq_api_key_set: bool
    ollama_base_url: str
    ollama_model: str
    embedding_model: str
    judge_threshold: float
    crag_threshold: float
    max_revisions: int


class OllamaModel(BaseModel):
    name: str
    size_gb: float
    family: str


class OllamaModelsResponse(BaseModel):
    available: bool
    base_url: str
    models: list[OllamaModel]
    error: Optional[str] = None


class GroqModelsResponse(BaseModel):
    models: list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

_GROQ_MODELS = [
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gemma-7b-it",
]

_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
]


def _read_env() -> dict[str, str]:
    """Parse .env into a dict."""
    env: dict[str, str] = {}
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def _write_env(updates: dict[str, str]) -> None:
    """Write key=value updates into .env, preserving comments and order."""
    _ENV_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    if _ENV_PATH.exists():
        lines = _ENV_PATH.read_text(encoding="utf-8").splitlines()

    written: set[str] = set()
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            new_lines.append(line)
            continue
        if "=" in stripped:
            k = stripped.split("=")[0].strip()
            if k in updates:
                new_lines.append(f"{k}={updates[k]}")
                written.add(k)
                continue
        new_lines.append(line)

    for k, v in updates.items():
        if k not in written:
            new_lines.append(f"{k}={v}")

    _ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _reload_settings() -> None:
    """Reload config singleton so the new values take effect immediately."""
    import importlib
    import config as cfg_module
    get_settings = cfg_module.get_settings
    get_settings.cache_clear()
    importlib.reload(cfg_module)
    cfg_module.settings = cfg_module.get_settings()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/config", response_model=LLMConfigResponse)
async def get_config() -> LLMConfigResponse:
    """Return current LLM + embedding configuration."""
    env = _read_env()
    groq_key = env.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    return LLMConfigResponse(
        provider=env.get("LLM_PROVIDER", "groq"),
        groq_model=env.get("GROQ_MODEL", "llama3-70b-8192"),
        groq_api_key_set=bool(groq_key and groq_key != "your-groq-api-key-here"),
        ollama_base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=env.get("OLLAMA_MODEL", "llama3"),
        embedding_model=env.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        judge_threshold=float(env.get("JUDGE_THRESHOLD", "0.7")),
        crag_threshold=float(env.get("CRAG_THRESHOLD", "0.5")),
        max_revisions=int(env.get("MAX_REVISIONS", "3")),
    )


@router.post("/config", response_model=LLMConfigResponse)
async def save_config(payload: LLMConfig) -> LLMConfigResponse:
    """
    Persist LLM + embedding configuration to .env and hot-reload settings.
    All subsequent LLM calls will use the new configuration.
    """
    updates: dict[str, str] = {
        "LLM_PROVIDER": payload.provider,
        "GROQ_MODEL": payload.groq_model,
        "OLLAMA_BASE_URL": payload.ollama_base_url,
        "OLLAMA_MODEL": payload.ollama_model,
        "EMBEDDING_MODEL": payload.embedding_model,
        "JUDGE_THRESHOLD": str(payload.judge_threshold),
        "CRAG_THRESHOLD": str(payload.crag_threshold),
        "MAX_REVISIONS": str(payload.max_revisions),
    }
    if payload.groq_api_key:
        updates["GROQ_API_KEY"] = payload.groq_api_key

    _write_env(updates)

    for k, v in updates.items():
        os.environ[k] = v

    try:
        _reload_settings()
    except Exception:
        pass

    return await get_config()


@router.get("/config/ollama/models", response_model=OllamaModelsResponse)
async def get_ollama_models(base_url: str = "http://localhost:11434") -> OllamaModelsResponse:
    """
    Query the local Ollama server for installed models.
    Returns the list with name, size, and family metadata.
    """
    candidates = [base_url.rstrip("/")]
    if "localhost" in candidates[0]:
        candidates.append(candidates[0].replace("localhost", "127.0.0.1"))
    elif "127.0.0.1" in candidates[0]:
        candidates.append(candidates[0].replace("127.0.0.1", "localhost"))

    last_error = ""
    for url in dict.fromkeys(candidates):
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(f"{url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
            base_url = url
            break
        except httpx.ConnectError:
            last_error = f"Cannot connect to Ollama at {url}. Is `ollama serve` running?"
        except httpx.HTTPStatusError as exc:
            last_error = f"Ollama API error at {url}: {exc.response.status_code}"
        except Exception as exc:
            last_error = str(exc)
    else:
        return OllamaModelsResponse(
            available=False,
            base_url=base_url.rstrip("/"),
            models=[],
            error=last_error,
        )

    raw_models = data.get("models", [])
    models = []
    for m in raw_models:
        name = m.get("name", "")
        size_bytes = m.get("size", 0)
        details = m.get("details", {})
        family = details.get("family", details.get("families", ["unknown"])[0] if details.get("families") else "unknown")
        models.append(OllamaModel(
            name=name,
            size_gb=round(size_bytes / 1e9, 2),
            family=str(family),
        ))

    return OllamaModelsResponse(
        available=True,
        base_url=base_url,
        models=models,
    )


# [image-based-campaign] Expose the image feature flag so the UI can show/hide
# the image uploader and generated-image panels.
@router.get("/config/features")
async def get_features() -> dict[str, bool]:
    from config import settings as _settings
    return {"image_feature_enabled": bool(_settings.IMAGE_FEATURE_ENABLED)}


@router.get("/config/groq/models", response_model=GroqModelsResponse)
async def get_groq_models() -> GroqModelsResponse:
    """Return the list of supported Groq models."""
    return GroqModelsResponse(models=_GROQ_MODELS)


@router.get("/config/embedding/models")
async def get_embedding_models() -> dict:
    """Return recommended SentenceTransformer embedding models."""
    return {"models": _EMBEDDING_MODELS}


@router.post("/config/test")
async def test_connection(payload: LLMConfig) -> dict[str, Any]:
    """
    Test the LLM connection without saving.
    Returns success/error and latency_ms.
    """
    import time

    if payload.provider == "groq":
        if not payload.groq_api_key:
            return {"success": False, "error": "GROQ_API_KEY is required to test Groq."}
        try:
            from langchain_groq import ChatGroq
            t0 = time.monotonic()
            llm = ChatGroq(api_key=payload.groq_api_key, model=payload.groq_model, temperature=0)
            resp = llm.invoke("Reply with one word: hello")
            ms = int((time.monotonic() - t0) * 1000)
            return {"success": True, "provider": "groq", "model": payload.groq_model,
                    "latency_ms": ms, "response_preview": str(resp.content)[:80]}
        except Exception as exc:
            return {"success": False, "provider": "groq", "error": str(exc)}

    if payload.provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            t0 = time.monotonic()
            llm = ChatOllama(base_url=payload.ollama_base_url, model=payload.ollama_model, temperature=0)
            resp = llm.invoke("Reply with one word: hello")
            ms = int((time.monotonic() - t0) * 1000)
            return {"success": True, "provider": "ollama", "model": payload.ollama_model,
                    "latency_ms": ms, "response_preview": str(resp.content)[:80]}
        except Exception as exc:
            return {"success": False, "provider": "ollama", "error": str(exc)}

    return {"success": False, "error": f"Unknown provider: {payload.provider}"}
