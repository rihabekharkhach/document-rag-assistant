# app/agents/retrieve.py
from app.agents.state import GraphState
from app.rag.vector_store import retrieve_chunks



def retrieve_node(state: GraphState) -> dict:
    chunks = retrieve_chunks(state["standalone_query"], top_k=20, sources=state.get("sources"))
    return {"chunks": chunks}