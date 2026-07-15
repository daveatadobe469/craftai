from __future__ import annotations

import asyncio
import logging
import math
import sys
from typing import Any
from unittest.mock import MagicMock

from config import get_llm, settings

logger = logging.getLogger(__name__)

_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _ensure_ragas_importable() -> None:
    """
    ragas 0.4.x imports Vertex AI from langchain-community paths removed in 0.4.x.
    Stub them so evaluation can load without optional Google Cloud packages.
    """
    sys.modules.setdefault(
        "langchain_community.chat_models.vertexai",
        MagicMock(),
    )
    sys.modules.setdefault(
        "langchain_community.llms.vertexai",
        MagicMock(),
    )


def _zero_scores() -> dict[str, float]:
    return {name: 0.0 for name in _METRICS}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _extract_scores(result: Any) -> dict[str, float]:
    """Parse ragas EvaluationResult (0.4+) or legacy dict/DataFrame outputs."""
    scores = _zero_scores()

    if hasattr(result, "_repr_dict"):
        for name in _METRICS:
            val = _safe_float(getattr(result, "_repr_dict", {}).get(name))
            if val is not None:
                scores[name] = val
        return scores

    if hasattr(result, "to_pandas"):
        try:
            df = result.to_pandas()
            for name in _METRICS:
                if name in df.columns:
                    val = _safe_float(df[name].iloc[0])
                    if val is not None:
                        scores[name] = val
            return scores
        except Exception:
            pass

    if isinstance(result, dict):
        for name in _METRICS:
            val = _safe_float(result.get(name))
            if val is not None:
                scores[name] = val
        return scores

    for name in _METRICS:
        try:
            series = result[name]
            val = _safe_float(series[0] if hasattr(series, "__getitem__") else series)
            if val is not None:
                scores[name] = val
        except Exception:
            continue

    return scores


class _LocalSentenceTransformerEmbeddings:
    """LangChain-compatible wrapper around the project's embedder module."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from rag import embedder

        return embedder.encode(texts)

    def embed_query(self, text: str) -> list[float]:
        from rag import embedder

        return embedder.encode_single(text)


def _build_ragas_llm():
    """Use the same provider as the rest of CRAFTAI (Groq or Ollama)."""
    if settings.LLM_PROVIDER == "groq" and not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required for RAGAS when LLM_PROVIDER=groq.")
    return get_llm(temperature=0)


async def evaluate_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str = "",
) -> dict[str, float]:
    """
    Run RAGAS evaluation on a single Q&A + retrieved contexts tuple.

    Returns faithfulness, answer_relevancy, context_precision, context_recall.
    On failure returns zeros and logs the underlying error.
    """
    if not answer.strip():
        logger.warning("RAGAS skipped: empty answer.")
        return _zero_scores()

    try:
        _ensure_ragas_importable()

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        llm = _build_ragas_llm()
        embeddings = _LocalSentenceTransformerEmbeddings()

        data: dict[str, list[Any]] = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts if contexts else ["No context retrieved."]],
            "ground_truth": [ground_truth or answer],
        }
        dataset = Dataset.from_dict(data)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: evaluate(
                dataset,
                metrics=[
                    Faithfulness(),
                    AnswerRelevancy(),
                    ContextPrecision(),
                    ContextRecall(),
                ],
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=False,
                show_progress=False,
            ),
        )
        scores = _extract_scores(result)
        if all(v == 0.0 for v in scores.values()):
            logger.warning(
                "RAGAS returned all zeros � check LLM provider (%s), API keys, and logs.",
                settings.LLM_PROVIDER,
            )
        else:
            logger.info("RAGAS scores: %s", scores)
        return scores

    except Exception as exc:
        logger.warning("RAGAS evaluation failed: %s", exc, exc_info=True)
        return _zero_scores()
