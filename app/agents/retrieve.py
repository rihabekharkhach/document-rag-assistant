# app/agents/retrieve.py
from app.agents.state import GraphState
from app.rag.vector_store import retrieve_chunks


def retrieve_node(state: GraphState) -> dict:
    return {"chunks": retrieve_chunks(state["query"], top_k=5)}