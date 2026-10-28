from app.agents.state import GraphState
from app.rag.reranker import rerank_indices
import os


def rerank_node(state: GraphState) -> dict:
    order = rerank_indices(state["standalone_query"], state["chunks"], top_n=int(os.getenv("FINAL_CHUNKS", "5")))
    items = [state["items"][i] for i in order]          # reordered metadata
    return {"items": items, "chunks": [item["text"] for item in items]}