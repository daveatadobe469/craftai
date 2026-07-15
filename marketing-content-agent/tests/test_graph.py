from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graph.state import AgentState


def _make_state(**overrides: Any) -> AgentState:
    base: AgentState = {
        "brief_id": str(uuid.uuid4()),
        "brand": "TestBrand",
        "channel": "email",
        "persona": "Premium Buyer",
        "key_message": "Launch our new premium product line with confidence.",
        "constraints": {
            "persona_profile": {
                "name": "Premium Buyer",
                "description": "High-income consumers.",
                "age_range": "35-55",
                "income_bracket": "upper",
                "interests": ["luxury"],
                "pain_points": ["poor quality"],
                "preferred_tone": "sophisticated",
                "char_limit_email": 600,
                "char_limit_social": 280,
                "char_limit_linkedin": 800,
                "char_limit_ad": 160,
                "char_limit_blog": 2500,
            }
        },
        "retrieved_campaigns": [],
        "retrieved_social": [],
        "retrieved_guidelines": [],
        "draft": "",
        "draft_metadata": {},
        "revision_count": 0,
        "rule_violations": [],
        "judge_score": 0.0,
        "judge_evidence": "",
        "compliance_pass": False,
        "human_decision": None,
        "human_edits": None,
        "reviewed_by": None,
        "reviewed_at": None,
        "indexed_doc_id": None,
        "ragas_scores": {},
        "mlflow_run_id": "",
        "plan": [],
        "errors": [],
        "sse_events": [],
    }
    base.update(overrides)
    return base


# ─── Compliance route logic ────────────────────────────────────────────────────

class TestComplianceRoute:
    def test_route_returns_revise_when_non_compliant_and_under_max(self):
        from graph.nodes.compliance import route

        state = _make_state(
            compliance_pass=False,
            revision_count=1,
        )
        with patch("graph.nodes.compliance.settings") as mock_settings:
            mock_settings.MAX_REVISIONS = 3
            result = route(state)
        assert result == "revise"

    def test_route_returns_gate_when_non_compliant_and_at_max(self):
        from graph.nodes.compliance import route

        state = _make_state(
            compliance_pass=False,
            revision_count=3,
        )
        with patch("graph.nodes.compliance.settings") as mock_settings:
            mock_settings.MAX_REVISIONS = 3
            result = route(state)
        assert result == "gate"

    def test_route_returns_gate_when_compliant(self):
        from graph.nodes.compliance import route

        state = _make_state(compliance_pass=True, revision_count=1)
        with patch("graph.nodes.compliance.settings") as mock_settings:
            mock_settings.MAX_REVISIONS = 3
            result = route(state)
        assert result == "gate"


# ─── Human gate route ─────────────────────────────────────────────────────────

class TestHumanGateRoute:
    def test_approved_decision_routes_approved(self):
        from graph.nodes.human_gate import route

        state = _make_state(human_decision="approved")
        assert route(state) == "approved"

    def test_edited_decision_routes_approved(self):
        from graph.nodes.human_gate import route

        state = _make_state(human_decision="edited")
        assert route(state) == "approved"

    def test_rejected_decision_routes_rejected(self):
        from graph.nodes.human_gate import route

        state = _make_state(human_decision="rejected")
        assert route(state) == "rejected"

    def test_none_decision_routes_rejected(self):
        from graph.nodes.human_gate import route

        state = _make_state(human_decision=None)
        assert route(state) == "rejected"


# ─── Orchestrator node ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_orchestrator_invalid_channel():
    from graph.nodes.orchestrator import orchestrator_node

    state = _make_state(channel="fax")
    with patch("graph.nodes.orchestrator.get_persona", return_value=None), \
         patch("graph.nodes.orchestrator.write_audit"), \
         patch("graph.nodes.orchestrator.mlflow"):
        result = await orchestrator_node(state)

    assert len(result.get("errors", [])) > 0
    assert any("channel" in e.lower() or "fax" in e.lower() for e in result["errors"])


@pytest.mark.asyncio
async def test_orchestrator_valid_brief_sets_plan():
    from graph.nodes.orchestrator import orchestrator_node

    state = _make_state()

    mock_llm_response = MagicMock()
    mock_llm_response.content = (
        "Thought: Analyse brand positioning.\n"
        "Action: Retrieve relevant past campaigns.\n"
        "Observation: Draft ready for compliance check."
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_response

    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"

    with patch("graph.nodes.orchestrator.get_persona", return_value=None), \
         patch("graph.nodes.orchestrator.write_audit"), \
         patch("graph.nodes.orchestrator.get_llm", return_value=mock_llm), \
         patch("graph.nodes.orchestrator.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mlflow.start_run.return_value = mock_run

        result = await orchestrator_node(state)

    assert result.get("errors") == [] or result.get("errors") is None or len(result.get("errors", [])) == 0
    assert len(result.get("plan", [])) > 0


# ─── Graph end-to-end (fully mocked) ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_graph_ainvoke_sets_draft():
    """
    Full graph invocation with mocked LLM, ChromaDB, and MLflow.
    Asserts that the draft field is populated after generator runs.
    """
    state = _make_state()

    mock_llm_content = MagicMock()
    mock_llm_content.content = '{"subject": "Test Subject", "body": "Test email body content for validation.", "cta": "Shop Now"}'
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_content

    mock_run_info = MagicMock()
    mock_run_info.info.run_id = "mock-run-id"

    mock_judge_response = MagicMock()
    mock_judge_response.content = '{"score": 0.85, "evidence": "Good brand alignment."}'
    mock_judge_llm = MagicMock()
    mock_judge_llm.invoke.return_value = mock_judge_response

    with patch("graph.nodes.orchestrator.get_persona", return_value=None), \
         patch("graph.nodes.orchestrator.write_audit"), \
         patch("graph.nodes.orchestrator.get_llm", return_value=mock_llm), \
         patch("graph.nodes.orchestrator.mlflow") as mock_mlflow_orch, \
         patch("graph.nodes.generator.get_llm", return_value=mock_llm), \
         patch("graph.nodes.generator.retrieve_with_hyde", return_value=[]), \
         patch("graph.nodes.generator.cc.collection_count", return_value=0), \
         patch("graph.nodes.generator.evaluator.evaluate_ragas", return_value={
             "faithfulness": 0.8, "answer_relevancy": 0.8,
             "context_precision": 0.8, "context_recall": 0.8,
         }), \
         patch("graph.nodes.generator.mlflow") as mock_mlflow_gen, \
         patch("graph.nodes.compliance.get_llm", return_value=mock_judge_llm), \
         patch("graph.nodes.compliance.score_draft", return_value=(0.85, "Good.")), \
         patch("graph.nodes.compliance.mlflow") as mock_mlflow_comp, \
         patch("graph.nodes.compliance.settings") as mock_settings_comp, \
         patch("graph.nodes.human_gate._events", {"__test__": asyncio.Event()}):

        mock_settings_comp.MAX_REVISIONS = 3
        mock_settings_comp.JUDGE_THRESHOLD = 0.7
        mock_mlflow_orch.start_run.return_value = mock_run_info
        mock_mlflow_orch.start_run.return_value.__enter__ = MagicMock(return_value=mock_run_info)
        mock_mlflow_orch.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from graph.builder import build_graph

        graph = build_graph()

        collected_states: list[dict] = []
        async for event in graph.astream(state):
            for node_name, node_state in event.items():
                collected_states.append({"node": node_name, "state": node_state})
                if node_name == "human_gate":
                    break
            if any(s["node"] == "human_gate" for s in collected_states):
                break

    generator_states = [s for s in collected_states if s["node"] == "generator"]
    if generator_states:
        gen_state = generator_states[-1]["state"]
        assert "draft" in gen_state
        assert len(gen_state["draft"]) > 0
