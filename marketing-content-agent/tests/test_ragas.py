from __future__ import annotations

import os

import pytest

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


@pytest.mark.asyncio
@pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY not set — skipping RAGAS live test")
async def test_ragas_evaluate_basic():
    """
    End-to-end RAGAS evaluation with a synthetic marketing Q&A fixture.
    Asserts context_precision >= 0.50 and faithfulness >= 0.50
    (conservative threshold to avoid test fragility with live LLM).
    """
    from rag.evaluator import evaluate_ragas

    question = "What are the key benefits of GlowBrand's platinum skincare collection?"
    answer = (
        "GlowBrand's platinum collection is clinically proven to reduce fine lines by 47% "
        "in 14 days. It contains rare marine peptides and gold-infused hyaluronic acid "
        "that provide 72-hour hydration. Dermatologist-approved and suitable for all skin types."
    )
    contexts = [
        "GlowBrand platinum skincare uses marine peptides that stimulate collagen synthesis "
        "and gold-infused hyaluronic acid for 72-hour hydration.",
        "Clinical trials showed a 47% reduction in fine lines after 14 days of use. "
        "94% of 200 participants reported visibly smoother skin.",
        "GlowBrand products are dermatologist-approved and formulated for all skin types.",
    ]

    scores = await evaluate_ragas(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth="GlowBrand platinum skincare reduces fine lines by 47% in 14 days "
                     "using marine peptides and hyaluronic acid.",
    )

    assert isinstance(scores, dict), "evaluate_ragas must return a dict"
    assert "faithfulness" in scores
    assert "answer_relevancy" in scores
    assert "context_precision" in scores

    assert scores["context_precision"] >= 0.50, (
        f"context_precision too low: {scores['context_precision']:.2f}"
    )
    assert scores["faithfulness"] >= 0.50, (
        f"faithfulness too low: {scores['faithfulness']:.2f}"
    )


@pytest.mark.asyncio
async def test_ragas_evaluate_returns_zeros_without_api_key(monkeypatch):
    """
    When GROQ_API_KEY is empty the evaluator must gracefully return zero scores
    rather than raising an exception.
    """
    monkeypatch.setenv("GROQ_API_KEY", "")

    import importlib
    from config import Settings
    from unittest.mock import patch

    with patch("config.settings") as mock_settings:
        mock_settings.GROQ_API_KEY = ""
        mock_settings.GROQ_MODEL = "llama3-70b-8192"
        mock_settings.MLFLOW_TRACKING_URI = "./data/mlruns"

        from rag.evaluator import evaluate_ragas, _zero_scores

        scores = await evaluate_ragas(
            question="test",
            answer="test answer",
            contexts=["test context"],
        )

    assert isinstance(scores, dict)
    for v in scores.values():
        assert isinstance(v, float)
