from __future__ import annotations

import asyncio
from typing import Any

import mlflow

from api.sse_queues import push_sync
from compliance.judge import score_draft
from compliance.tools import run_all_checks
from config import get_judge_llm, settings
from db.sqlite import update_brief_status, write_draft
from graph.state import AgentState


def route(state: AgentState) -> str:
    """
    Conditional edge router after compliance_node.
    Returns "revise" if non-compliant and under MAX_REVISIONS, else "gate".
    """
    # Fail open: if the judge could not be evaluated (rate limit / API error) we
    # cannot claim the draft is non-compliant, so send it to a human instead of
    # burning revisions on a score that was never real.
    if state.get("judge_unavailable"):
        return "gate"

    revision_count = state.get("revision_count") or 0
    compliance_pass = state.get("compliance_pass", False)

    if not compliance_pass and revision_count < settings.MAX_REVISIONS:
        return "revise"
    return "gate"


def _ragas_contexts(state: AgentState) -> list[str]:
    """Retrieved documents used as RAGAS context for the final draft."""
    retrieved = (state.get("retrieved_campaigns") or []) + (state.get("retrieved_guidelines") or [])
    return [c.get("document", "") for c in retrieved if c.get("document")]


# Sentinel recorded instead of RAGAS scores when there is no retrieved context to
# ground against — scoring a draft against an empty/placeholder context always
# yields hard zeros, which reads as a broken metric rather than "not applicable".
RAGAS_NO_CONTEXT = {"not_evaluated": "no grounding context retrieved"}


async def _run_ragas(state: AgentState, draft: str) -> dict[str, Any]:
    """Single RAGAS evaluation of the draft heading to the human gate.

    Skipped when retrieval returned no context: RAGAS's faithfulness / context
    metrics are undefined without contexts, so we record a not-evaluated sentinel
    instead of misleading zeros.
    """
    from rag import evaluator

    contexts = _ragas_contexts(state)
    if not contexts:
        return dict(RAGAS_NO_CONTEXT)

    brief_text = (
        f"Brand: {state.get('brand', '')}. Channel: {state.get('channel', '')}. "
        f"Persona: {state.get('persona', '')}. Key message: {state.get('key_message', '')}."
    )
    return await evaluator.evaluate_ragas(
        question=brief_text,
        answer=draft,
        contexts=contexts,
    )


async def compliance_node(state: AgentState) -> AgentState:
    """
    Stage 3 — Compliance Node
    Pass 1: Deterministic rule tools (no LLM).
    Pass 2: LLM-as-judge for holistic brand alignment.
    Sets compliance_pass based on violations + judge_score.
    """
    errors: list[str] = list(state.get("errors") or [])
    sse_events: list[str] = list(state.get("sse_events") or [])

    try:
        brief_id = state["brief_id"]
        draft = state.get("draft", "")
        channel = state["channel"]
        brand = state["brand"]
        constraints = state.get("constraints") or {}
        revision_count = state.get("revision_count") or 0

        # [ring-liveness] Emit progress the instant it happens, not when the node
        # returns. LangGraph hands a node's batched sse_events to the client only
        # on RETURN, so the judge + RAGAS work (20–40s) would otherwise leave the
        # UI progress ring frozen. push_sync writes straight to the live SSE queue;
        # these messages are delivered here instead of via the node-end flush, so
        # there is no duplication (they are not appended to sse_events).
        def live(msg: str) -> None:
            push_sync(brief_id, msg)

        if not draft:
            errors.append("Compliance: no draft to check.")
            # [rag-perf] Surface WHY there is no draft. A bare "No draft
            # generated." reads as a content failure, when the real cause is
            # almost always upstream (API rate limit, empty completion) and is
            # already recorded in errors by the generator.
            upstream = next(
                (e for e in reversed(errors) if e.startswith("Generator")), ""
            )
            reason = (
                f"No draft generated — {upstream}" if upstream else "No draft generated."
            )
            # Count this as a revision. route() sends "not passed and under
            # MAX_REVISIONS" back to the generator, so leaving the counter
            # untouched here made a generator that cannot produce a draft loop
            # forever instead of reaching a human.
            new_rc = revision_count + 1
            terminal = new_rc >= settings.MAX_REVISIONS

            # [failure-visibility] When we're out of retries, PERSIST the failure.
            # The success path below is the only place that writes a draft row +
            # flips status off "pending"; the no-draft path used to return without
            # either, so a generation failure (rate limit / empty completion)
            # routed to the gate but left the brief frozen at "pending" with no
            # draft and no visible error — indistinguishable from a hang. Write a
            # clearly-marked failure draft and surface it for review instead.
            if terminal:
                live(f"[Compliance] {reason}")
                fail_meta = {
                    "generation_failed": True,
                    "failure_reason": reason,
                    "rule_violations": [reason],
                }
                fail_content = f"⚠️ Generation failed — no draft was produced.\n\n{reason}"
                try:
                    lp = asyncio.get_event_loop()
                    await lp.run_in_executor(
                        None,
                        lambda: write_draft(
                            brief_id=brief_id,
                            content=fail_content,
                            revision_count=new_rc,
                            metadata=fail_meta,
                            judge_score=0.0,
                            compliance_pass=False,
                            ragas_scores={},
                            mlflow_run_id=state.get("mlflow_run_id") or "",
                        ),
                    )
                    await lp.run_in_executor(
                        None, update_brief_status, brief_id, "awaiting_review",
                    )
                except Exception as db_exc:  # noqa: BLE001 — never mask the real failure
                    errors.append(f"Compliance DB write error (no-draft): {db_exc}")

            return {
                **state,
                "compliance_pass": False,
                "rule_violations": [reason],
                "judge_score": 0.0,
                "judge_evidence": reason,
                "revision_count": new_rc,
                "errors": errors,
                "sse_events": sse_events,
            }

        live("[Compliance] Running deterministic rule checks…")

        persona_profile = constraints.get("persona_profile", {})
        persona_limits: dict[str, int] = {
            k: v for k, v in persona_profile.items() if k.startswith("char_limit_")
        }

        violations = run_all_checks(
            draft=draft,
            channel=channel,
            brand=brand,
            persona_limits=persona_limits if persona_limits else None,
            extra_required=None,
            allowed_domains=None,
            draft_metadata=state.get("draft_metadata") or {},
            require_email_footer=True,
        )

        live(f"[Compliance] Rule check complete — {len(violations)} violation(s) found.")

        retrieved_guidelines = state.get("retrieved_guidelines") or []
        guidelines_context = "\n".join(
            g.get("document", "") for g in retrieved_guidelines[:3]
        )

        llm = get_judge_llm(temperature=0.1)
        live("[Compliance] Running LLM-as-judge scoring…")

        loop = asyncio.get_event_loop()
        raw_score, judge_evidence = await score_draft(
            draft=draft,
            channel=channel,
            brand=brand,
            guidelines_context=guidelines_context,
            llm=llm,
        )

        # score_draft returns None when the judge could not run at all.
        judge_unavailable = raw_score is None
        judge_score = 0.0 if judge_unavailable else raw_score

        if judge_unavailable:
            compliance_pass = False
            live(
                f"[Compliance] Judge UNAVAILABLE ({judge_evidence}). "
                "Failing open — sending to human review without counting a revision."
            )
        else:
            compliance_pass = (
                len(violations) == 0 and judge_score >= settings.JUDGE_THRESHOLD
            )
            live(
                f"[Compliance] Judge score: {judge_score:.2f} "
                f"(threshold: {settings.JUDGE_THRESHOLD}). "
                f"Pass: {compliance_pass}."
            )

        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("craftai_campaigns")
            with mlflow.start_run(run_name=brief_id, nested=False):
                mlflow.log_metrics({
                    "judge_score": judge_score,
                    "rule_violations_count": len(violations),
                    "compliance_pass": int(compliance_pass),
                })
        except Exception:
            pass

        # Don't count a revision when the judge never actually ran.
        if judge_unavailable:
            new_revision_count = revision_count
        else:
            new_revision_count = revision_count + (0 if compliance_pass else 1)

        # Persist draft to SQLite NOW so the status endpoint can read it
        # at the human-gate stage (before the curator runs after approval).
        will_gate = (
            judge_unavailable
            or compliance_pass
            or (new_revision_count >= settings.MAX_REVISIONS)
        )

        # RAGAS runs ONCE, here, on the draft that is actually going to a human —
        # not on every revision (that was ~4 LLM calls per loop and exhausted quota).
        ragas_scores = state.get("ragas_scores") or {}
        if will_gate and settings.RAGAS_ENABLED and not ragas_scores:
            live("[Compliance] Running RAGAS on final draft…")
            ragas_scores = await _run_ragas(state, draft)
            if "not_evaluated" in ragas_scores:
                live("[Compliance] RAGAS not evaluated — no grounding context retrieved.")
            else:
                live(
                    f"[Compliance] RAGAS — "
                    f"faithfulness: {ragas_scores.get('faithfulness', 0):.2f}, "
                    f"relevancy: {ragas_scores.get('answer_relevancy', 0):.2f}"
                )
        # Merge violations + judge evidence into metadata so status endpoint
        # can surface them without a DB schema change.
        draft_metadata = dict(state.get("draft_metadata") or {})
        draft_metadata["rule_violations"]  = violations
        draft_metadata["judge_evidence"]   = judge_evidence

        try:
            await loop.run_in_executor(
                None,
                lambda: write_draft(
                    brief_id=brief_id,
                    content=draft,
                    revision_count=revision_count,
                    metadata=draft_metadata,
                    judge_score=judge_score,
                    compliance_pass=compliance_pass,
                    ragas_scores=ragas_scores,
                    mlflow_run_id=state.get("mlflow_run_id") or "",
                ),
            )
            await loop.run_in_executor(
                None,
                update_brief_status,
                brief_id,
                "awaiting_review" if will_gate else "processing",
            )
        except Exception as db_exc:
            errors.append(f"Compliance DB write error: {db_exc}")

        return {
            **state,
            "rule_violations": violations,
            "judge_score": judge_score,
            "judge_evidence": judge_evidence,
            "judge_unavailable": judge_unavailable,
            "compliance_pass": compliance_pass,
            "revision_count": new_revision_count,
            "ragas_scores": ragas_scores,
            "draft_metadata": draft_metadata,
            "errors": errors,
            "sse_events": sse_events,
        }

    except Exception as exc:
        errors.append(f"Compliance error: {exc}")
        sse_events.append(f"[Compliance] ERROR: {exc}")
        # [rag-perf] Same reasoning as the no-draft branch above: this path also
        # routes back to the generator, so the counter must advance or a repeated
        # internal error becomes an unbounded generator/compliance loop.
        return {
            **state,
            "compliance_pass": False,
            "judge_unavailable": False,
            "rule_violations": [f"Internal error: {exc}"],
            "judge_score": 0.0,
            "judge_evidence": f"error: {exc}",
            "revision_count": (state.get("revision_count") or 0) + 1,
            "errors": errors,
            "sse_events": sse_events,
        }
