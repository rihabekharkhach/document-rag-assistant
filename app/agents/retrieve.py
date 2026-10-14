# app/agents/retrieve.py
from app.agents.state import GraphState
from app.rag.vector_store import retrieve_chunks



def retrieve_node(state: GraphState) -> dict:
    question = state["standalone_query"]
    return {"chunks": retrieve_chunks(question, top_k=20)}