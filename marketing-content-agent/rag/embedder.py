from __future__ import annotations

import threading
from typing import Any

from sentence_transformers import SentenceTransformer

from config import settings

_lock = threading.Lock()
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def encode(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Encode a list of texts into embedding vectors.

    Returns a list of float lists (one per input text).
    """
    if not texts:
        return []
    model = get_model()
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
