# app/main.py

"""
    python -m uvicorn app.main:app --reload     then open http://127.0.0.1:8000/docs
"""

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
import pymupdf

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from app import db
from app.agents.graph import research_graph
from app.models import (
    ConversationOut, DocumentOut, IngestReportOut, MessageOut, QueryRequest, QueryResponse, RenameIn
)
from app.rag.ingest import SOURCE_DIR, ingest_file
from app.rag.vector_store import delete_source, list_sources

logger = logging.getLogger("rag-api")

ALLOWED_SUFFIXES = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
HISTORY_MESSAGES = 6


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    Path(SOURCE_DIR).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Conversational RAG API", version="0.3.0", lifespan=lifespan)  # <- app is created here


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


@app.get("/image")
def image(path: str) -> FileResponse:
    """Serve a figure or an uploaded image. Only files inside the two data folders are allowed."""
    resolved = Path(path).resolve()
    allowed = [Path("data/figures").resolve(), Path(SOURCE_DIR).resolve()]
    if (
        not any(resolved.is_relative_to(folder) for folder in allowed)
        or resolved.suffix.lower() not in IMAGE_SUFFIXES
        or not resolved.is_file()
    ):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(resolved)


def _fragments(quote: str, window: int = 7, limit: int = 12) -> list[str]:
    """Short word windows of the quote: robust to line breaks and hyphenation in the PDF."""
    words = quote.split()
    return [" ".join(words[i:i + window]) for i in range(0, len(words), window)][:limit]


@app.get("/page-image")
def page_image(source: str, page: int, quote: str = "") -> Response:
    """Render one PDF page as an image, with the quoted passage highlighted when it can be found."""
    path = Path(SOURCE_DIR) / Path(source).name  # strips any folder part from the client's value
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found.")

    with pymupdf.open(path) as doc:
        if not 1 <= page <= len(doc):
            raise HTTPException(status_code=404, detail="Page not found.")
        pdf_page = doc[page - 1]
        for fragment in _fragments(quote):
            for rect in pdf_page.search_for(fragment):
                pdf_page.add_highlight_annot(rect)  # in memory only: the file on disk is never modified
        png = pdf_page.get_pixmap(dpi=130).tobytes("png")
    return Response(content=png, media_type="image/png")

# ---------- conversations ----------

@app.get("/conversations", response_model=list[ConversationOut])
def conversations() -> list[dict]:
    return db.list_conversations()


@app.patch("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, body: RenameIn) -> dict:
    if not db.rename_conversation(conversation_id, body.title):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"renamed": conversation_id}


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
        thread_id = db.create_conversation(title=(req.title or "").strip() or req.question)

    db.add_message(thread_id, "user", req.question)
    db.add_message(
        thread_id, "assistant", result["answer"],
        found=result["found"], citations=result["citations"],
        standalone_query=result["standalone_query"],
        sources=result["sources_used"],
        answer_mode=result.get("answer_mode", "text")
    )

    return QueryResponse(
        thread_id=thread_id,
        answer=result["answer"],
        found=result["found"],
        citations=result["citations"],
        standalone_query=result["standalone_query"],
        sources=result["sources_used"],
        answer_mode=result.get("answer_mode", "text")
    )