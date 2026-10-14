# app/agents/rerank.py
from app.agents.state import GraphState
from app.rag.reranker import rerank


def rerank_node(state: GraphState) -> dict:
    question = state["standalone_query"]
    return {"chunks": rerank(question, state["chunks"], top_n=4)}