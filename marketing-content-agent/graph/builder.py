from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graph.nodes.art_director import art_director_node  # [image-based-campaign]
from graph.nodes.compliance import compliance_node, route as compliance_route
from graph.nodes.curator import curator_node
from graph.nodes.generator import generator_node
from graph.nodes.human_gate import human_gate_node, route as gate_route
from graph.nodes.orchestrator import orchestrator_node
from graph.nodes.vision import vision_node  # [image-based-campaign]
from graph.state import AgentState


def build_graph():
    """
    Construct and compile the CRAFTAI LangGraph state machine.

    Pipeline ([image-based-campaign] adds vision + art_director; both are runtime
    no-ops when the brief has no uploaded image / generate_image is false):
        START
          → vision            (image → text, no-op if no image)
          → orchestrator
          → generator
          → compliance
          ↙ (revise)     ↘ (gate)
        generator       art_director   (text → image, no-op if disabled)
                          → human_gate
                          ↙ (approved)  ↘ (rejected)
                        curator          END
                          → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("vision", vision_node)  # [image-based-campaign]
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("generator", generator_node)
    graph.add_node("compliance", compliance_node)
    graph.add_node("art_director", art_director_node)  # [image-based-campaign]
    graph.add_node("human_gate", human_gate_node)
    graph.add_node("curator", curator_node)

    graph.add_edge(START, "vision")            # [image-based-campaign]
    graph.add_edge("vision", "orchestrator")   # [image-based-campaign]
    graph.add_edge("orchestrator", "generator")
    graph.add_edge("generator", "compliance")

    graph.add_conditional_edges(
        "compliance",
        compliance_route,
        {
            "revise": "generator",
            "gate": "art_director",  # [image-based-campaign] image before the gate
        },
    )
    graph.add_edge("art_director", "human_gate")  # [image-based-campaign]

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
