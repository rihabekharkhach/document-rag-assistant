# app/main.py

"""
    python -m uvicorn app.main:app --reload     then open http://127.0.0.1:8000/docs
"""

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app import db
from app.agents.graph import research_graph
from app.models import (
    ConversationOut, DocumentOut, IngestReportOut, MessageOut, QueryRequest, QueryResponse,
)
from app.rag.ingest import SOURCE_DIR, ingest_file
from app.rag.vector_store import delete_source, list_sources

logger = logging.getLogger("rag-api")

ALLOWED_SUFFIXES = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
HISTORY_MESSAGES = 6


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    Path(SOURCE_DIR).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Conversational RAG API", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ---------- documents ----------

@app.post("/documents", response_model=list[IngestReportOut])
def upload_documents(files: list[UploadFile] = File(...)) -> list[IngestReportOut]:
    """Upload one or several files; each is parsed, chunked and indexed. Returns one report per file."""
    reports: list[IngestReportOut] = []
    for upload in files:
        name = Path(upload.filename or "").name  # strips any folder part from the client's filename
        suffix = Path(name).suffix.lower()

        if suffix not in ALLOWED_SUFFIXES:
            reports.append(IngestReportOut(name=name or "?", status="error",
                                           message=f"Type de fichier non supporté ({suffix or 'inconnu'})."))
            continue

        data = upload.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            reports.append(IngestReportOut(name=name, status="error", message="Fichier trop volumineux (25 Mo max)."))
            continue

        path = Path(SOURCE_DIR) / name
        path.write_bytes(data)
        try:
            reports.append(IngestReportOut(**ingest_file(str(path))))
        except Exception:
            logger.exception("Ingestion failed for %s", name)
            reports.append(IngestReportOut(name=name, status="error",
                                           message="Échec de l'ingestion (voir les logs du serveur)."))
    return reports


@app.get("/documents", response_model=list[DocumentOut])
def documents() -> list[dict]:
    return list_sources()


@app.delete("/documents/{name}")
def remove_document(name: str) -> dict:
    name = Path(name).name
    delete_source(name)
    (Path(SOURCE_DIR) / name).unlink(missing_ok=True)
    shutil.rmtree(Path("data/figures") / Path(name).stem, ignore_errors=True)
    return {"deleted": name}


# ---------- conversations ----------

@app.get("/conversations", response_model=list[ConversationOut])
def conversations() -> list[dict]:
    return db.list_conversations()


@app.get("/conversations/{conversation_id}", response_model=list[MessageOut])
def conversation_messages(conversation_id: str) -> list[dict]:
    if not db.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return db.get_messages(conversation_id)


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str) -> dict:
    if not db.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"deleted": conversation_id}


# ---------- question answering ----------

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    thread_id = req.thread_id
    history: list[dict] = []

    if thread_id:
        if not db.conversation_exists(thread_id):
            raise HTTPException(status_code=404, detail="Conversation not found.")
        history = db.recent_history(thread_id, HISTORY_MESSAGES)

    try:
        result = research_graph.invoke(
            {"query": req.question, "chat_history": history, "sources": req.sources}
        )
    except Exception:
        logger.exception("Pipeline failed (thread_id=%s)", thread_id)
        raise HTTPException(status_code=502, detail="The pipeline failed. Check the server logs.")

    # Created only after success, so a failed first question leaves no empty conversation behind
    if not thread_id:
        thread_id = db.create_conversation(title=req.question)

    db.add_message(thread_id, "user", req.question)
    db.add_message(
        thread_id, "assistant", result["answer"],
        found=result["found"], citations=result["citations"],
        standalone_query=result["standalone_query"],
    )

    return QueryResponse(
        thread_id=thread_id,
        answer=result["answer"],
        found=result["found"],
        citations=result["citations"],
        standalone_query=result["standalone_query"],
    )