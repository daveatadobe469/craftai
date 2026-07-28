from __future__ import annotations

import os
import random
import threading
from typing import Any

# Must be set before sentence_transformers/huggingface_hub import: prevents any
# network call during model load (version/telemetry checks) that can hang
# indefinitely on restricted corporate networks, even when the model is
# already cached locally.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - import may fail in minimal environments
    SentenceTransformer = None  # type: ignore[assignment]

from config import settings

_lock = threading.Lock()
_model: Any | None = None


def _fallback_embedding(text: str) -> list[float]:
    """Simple deterministic fallback when the sentence-transformers model is unavailable."""
    rng = random.Random(hash(text) & 0xFFFFFFFF)
    return [round(rng.uniform(-1.0, 1.0), 6) for _ in range(384)]


def get_model() -> Any:
    """Lazy-load and cache the embedding model, with a deterministic offline fallback."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                if SentenceTransformer is None:
                    _model = "fallback"
                else:
                    try:
                        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
                    except Exception:
                        _model = "fallback"
    return _model


def encode(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Encode a list of texts into embedding vectors.

    Returns a list of float lists (one per input text).
    """
    if not texts:
        return []
    model = get_model()
    if model == "fallback":
        return [_fallback_embedding(text) for text in texts]

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vectors]


def encode_single(text: str) -> list[float]:
    """Convenience wrapper for encoding a single string."""
    return encode([text])[0]
