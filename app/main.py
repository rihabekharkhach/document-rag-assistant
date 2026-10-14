# app/main.py

"""
FastAPI entry point: exposes the RAG graph over HTTP.

Run from the project root:
    python -m uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs
"""

import logging
import uuid

from fastapi import FastAPI, HTTPException

from app.agents.graph import research_graph
from app.models import QueryRequest, QueryResponse

logger = logging.getLogger("rag-api")

app = FastAPI(title="Conversational RAG API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = research_graph.invoke({"query": req.question}, config=config)
    except Exception:
        # Full details go to the server log; the client gets a generic message.
        logger.exception("Pipeline failed (thread_id=%s)", thread_id)
        raise HTTPException(status_code=502, detail="The pipeline failed. Check the server logs.")

    return QueryResponse(
        thread_id=thread_id,
        answer=result["answer"],
        found=result["found"],
        citations=result["citations"],
        standalone_query=result["standalone_query"],
    )