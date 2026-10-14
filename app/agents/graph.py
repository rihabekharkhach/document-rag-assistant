# app/agents/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import GraphState
from app.agents.condense import condense_node
from app.agents.retrieve import retrieve_node
from app.agents.rerank import rerank_node
from app.agents.generate import generate_node


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("condense", condense_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("condense")
    graph.add_edge("condense", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=MemorySaver())


research_graph = build_graph()