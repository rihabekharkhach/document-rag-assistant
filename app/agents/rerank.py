# app/agents/rerank.py
from app.agents.state import GraphState
from app.rag.reranker import rerank


def rerank_node(state: GraphState) -> dict:
    return {"chunks": rerank(state["query"], state["chunks"], top_n=4)}