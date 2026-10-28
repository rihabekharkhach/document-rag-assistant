from app.agents.state import GraphState
from app.rag.vector_store import retrieve_items
import os 


def retrieve_node(state: GraphState) -> dict:
    top_k = int(os.getenv("RETRIEVE_CANDIDATES", "30"))
    items = retrieve_items(state["standalone_query"], top_k=top_k, sources=state.get("sources"))
    return {"items": items, "chunks": [item["text"] for item in items]}