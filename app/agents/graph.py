# app/agents/graph.py

from langgraph.graph import END, StateGraph

from app.agents.condense import condense_node
from app.agents.generate import generate_node
from app.agents.rerank import rerank_node
from app.agents.retrieve import retrieve_node
from app.agents.state import GraphState
from app.agents.vision import route_after_generate, vision_node


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("condense", condense_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("vision", vision_node)

    graph.set_entry_point("condense")
    graph.add_edge("condense", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_conditional_edges("generate", route_after_generate, {"vision": "vision", "end": END})
    graph.add_edge("vision", END)

    return graph.compile()


research_graph = build_graph()