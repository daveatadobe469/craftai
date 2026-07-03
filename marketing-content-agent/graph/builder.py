from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graph.nodes.compliance import compliance_node, route as compliance_route
from graph.nodes.curator import curator_node
from graph.nodes.generator import generator_node
from graph.nodes.human_gate import human_gate_node, route as gate_route
from graph.nodes.orchestrator import orchestrator_node
from graph.state import AgentState


def build_graph():
    """
    Construct and compile the CRAFTAI LangGraph state machine.

    Pipeline:
        START
          → orchestrator
          → generator
          → compliance
          ↙ (revise)     ↘ (gate)
        generator       human_gate
                          ↙ (approved)  ↘ (rejected)
                        curator          END
                          → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("generator", generator_node)
    graph.add_node("compliance", compliance_node)
    graph.add_node("human_gate", human_gate_node)
    graph.add_node("curator", curator_node)

    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "generator")
    graph.add_edge("generator", "compliance")

    graph.add_conditional_edges(
        "compliance",
        compliance_route,
        {
            "revise": "generator",
            "gate": "human_gate",
        },
    )

    graph.add_conditional_edges(
        "human_gate",
        gate_route,
        {
            "approved": "curator",
            "rejected": END,
        },
    )

    graph.add_edge("curator", END)

    return graph.compile()


compiled_graph = build_graph()
