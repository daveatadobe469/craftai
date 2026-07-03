from __future__ import annotations

import asyncio
from typing import Any

from config import settings


async def evaluate_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str = "",
) -> dict[str, float]:
    """
    Run RAGAS evaluation on a single Q&A + retrieved contexts tuple.

    Returns a dict with keys: faithfulness, answer_relevancy,
    context_precision, context_recall.  On any error returns zeros.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from langchain_groq import ChatGroq
        from langchain_core.embeddings import Embeddings

        if not settings.GROQ_API_KEY:
            return _zero_scores()

        llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0,
        )

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
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=llm,
                raise_exceptions=False,
            ),
        )
        scores: dict[str, float] = {}
        for metric_name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            val = result.get(metric_name)
            if val is None:
                scores[metric_name] = 0.0
            elif hasattr(val, "__float__"):
                scores[metric_name] = float(val)
            else:
                try:
                    scores[metric_name] = float(list(val)[0])
                except Exception:
                    scores[metric_name] = 0.0
        return scores

    except Exception:
        return _zero_scores()


def _zero_scores() -> dict[str, float]:
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
    }
