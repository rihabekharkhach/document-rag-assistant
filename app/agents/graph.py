# app/agents/graph.py

"""
Wires the Researcher, Critic, and Writer nodes into a LangGraph
StateGraph, including the conditional edge that creates the
loop between Researcher and Critic.
"""

from langgraph.graph import StateGraph, END

from app.agents.state import GraphState
from app.agents.researcher import researcher_node
from app.agents.critic import critic_node
from app.agents.writer import writer_node  # you'll write this one next


def route_after_critic(state: GraphState) -> str:
    """
    Conditional edge function: reads the state the Critic just
    updated, and returns the NAME of the next node to run.
    LangGraph calls this automatically after critic_node runs.
    """
    if state["is_sufficient"]:
        return "writer"
    return "researcher"


def build_graph():
    graph = StateGraph(GraphState)

    # Register each node under a name
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", writer_node)

    # Entry point: where the graph starts
    graph.set_entry_point("researcher")

    # Normal edge: researcher always goes to critic next
    graph.add_edge("researcher", "critic")

    # Conditional edge: after critic, branch based on route_after_critic()
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "researcher": "researcher",  # loop back
            "writer": "writer",          # proceed
        },
    )

    # Writer is the last step
    graph.add_edge("writer", END)

    return graph.compile()


# Compiled graph, ready to invoke
research_graph = build_graph()