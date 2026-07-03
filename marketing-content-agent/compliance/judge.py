from __future__ import annotations

import json
import re

from langchain_core.language_models import BaseChatModel

_FEW_SHOT_EXAMPLES = """
EXAMPLE 1
Draft: "Discover our exclusive range of premium skincare products. Transform your skin with clinically proven formulas. Shop now at brand.com/skincare"
Channel: email
Brand: GlowBrand
Score: 0.92
Evidence: "The draft aligns with premium brand positioning, avoids restricted claims, includes a clear CTA with an approved domain, and is appropriately concise for email."

EXAMPLE 2
Draft: "GUARANTEED RESULTS in just 7 days! No risk, instant results! Click here for your free money consultation. Limited time offer - act now!!!"
Channel: ad
Brand: HealthCo
Score: 0.08
Evidence: "The draft contains multiple restricted phrases (guaranteed, no risk, instant results, free money, click here, limited time offer, act now) and uses excessive punctuation indicative of spam."
"""

_SYSTEM_PROMPT = f"""You are a brand compliance judge evaluating marketing content.

Your task is to score a draft on a scale from 0.0 (completely non-compliant) to 1.0 (fully compliant).

Evaluate based on:
1. Brand voice and tone alignment
2. Absence of restricted/misleading language
3. Clarity and quality of the CTA
4. Appropriateness for the channel
5. Factual accuracy (no unsubstantiated superlatives)
6. Overall professional quality

{_FEW_SHOT_EXAMPLES}

Respond with a JSON object ONLY:
{{"score": <float 0.0-1.0>, "evidence": "<one paragraph explanation>"}}
"""


async def score_draft(
    draft: str,
    channel: str,
    brand: str,
    guidelines_context: str,
    llm: BaseChatModel,
) -> tuple[float, str]:
    """
    LLM-as-judge compliance scorer.

    Args:
        draft: The generated marketing content.
        channel: One of email, linkedin, social, ad, blog.
        brand: Brand name for tone alignment.
        guidelines_context: Relevant brand guideline chunks from ChromaDB.
        llm: A LangChain chat model instance.

    Returns:
        Tuple of (score: float, evidence: str).
        Returns (0.0, "parse error") on any failure.
    """
    import asyncio

    user_prompt = (
        f"Brand: {brand}\n"
        f"Channel: {channel}\n\n"
        f"Brand Guidelines Context:\n{guidelines_context or 'No guidelines retrieved.'}\n\n"
        f"Draft:\n{draft}\n\n"
        "Respond with JSON only."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, llm.invoke, messages)
        raw = response.content.strip()

        match = re.search(r'\{.*?"score"\s*:\s*([0-9.]+).*?"evidence"\s*:\s*"(.*?)".*?\}', raw, re.DOTALL)
        if match:
            score = min(1.0, max(0.0, float(match.group(1))))
            evidence = match.group(2).strip()
            return score, evidence

        data = json.loads(raw)
        score = min(1.0, max(0.0, float(data.get("score", 0.0))))
        evidence = str(data.get("evidence", ""))
        return score, evidence

    except json.JSONDecodeError:
        return 0.0, "parse error"
    except Exception as exc:
        return 0.0, f"judge error: {exc}"
