# app/agents/graph.py
from langgraph.graph import StateGraph, END

from app.agents.state import GraphState
from app.agents.retrieve import retrieve_node
from app.agents.generate import generate_node


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


research_graph = build_graph()