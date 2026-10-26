# app/rag/ingest.py

"""
Parse, chunk, embed and store documents. Each file returns a short report
(pages, characters, chunks, figures) that the UI shows right after upload.

    python -m app.rag.ingest          # ingest everything in data/source_documents
"""

import os
import time
from pathlib import Path

from app.rag.chunking import chunk_text
from app.rag.figures import Figure, extract_figures
from app.rag.loaders import load_file
from app.rag.ocr import ocr_image
from app.rag.vector_store import create_collection, delete_source, ingest_documents

SOURCE_DIR = "data/source_documents"
MAX_FIGURE_CHARS = 2000
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def figure_item(fig: Figure) -> dict:
    ocr_text = ocr_image(fig.image_path)[:MAX_FIGURE_CHARS]
    caption = fig.caption or "Figure (caption not found)"
    return {
        "text": f"[Figure, page {fig.page}] {caption}\nText detected in the figure:\n{ocr_text}",
        "type": "figure",
        "source": fig.source,
        "page": fig.page,
        "image_path": fig.image_path,
    }


def ingest_file(path: str) -> dict:
    """Ingest one file and return a report. Re-ingesting a file replaces its previous version."""
    start = time.time()
    name = Path(path).name
    create_collection()

    text_items: list[dict] = []
    figure_items: list[dict] = []
    warning = ""

    if name.lower().endswith(IMAGE_SUFFIXES):
        # A standalone image is ONE element: splitting OCR text would separate a schema's labels
        ocr_text = ocr_image(path)[:MAX_FIGURE_CHARS]
        figure_items = [{
            "text": f"[Image : {name}]\nText detected in the image:\n{ocr_text}",
            "type": "figure",
            "source": name,
            "page": 1,
            "image_path": path,
        }]
        pages, characters = 1, len(ocr_text)
        if len(ocr_text.strip()) < 20:
            warning = "Très peu de texte détecté dans l'image (photo, ou schéma sans texte ?)."
    else:
        raw_pages = load_file(path)  # raises ValueError for unsupported types
        text = "\n".join(raw_pages)
        text_items = [{"text": c, "type": "text", "source": name} for c in chunk_text(text)]
        if name.lower().endswith(".pdf"):
            figure_items = [figure_item(f) for f in extract_figures(path)]
        pages, characters = len(raw_pages), len(text)
        if len(text.strip()) < 50:
            warning = "Très peu de texte extrait : le document est peut-être un scan."

    items = text_items + figure_items
    if not items:
        return {"name": name, "status": "error", "message": "Aucun contenu exploitable trouvé."}

    delete_source(name)  # only after parsing succeeded, so a failed upload never wipes the old version
    ingest_documents(items)

    return {
        "name": name,
        "status": "ok",
        "pages": pages,
        "characters": characters,
        "text_chunks": len(text_items),
        "figures": len(figure_items),
        "seconds": round(time.time() - start, 1),
        "message": warning,
    }


def ingest_directory(source_dir: str) -> None:
    for filename in sorted(os.listdir(source_dir)):
        path = os.path.join(source_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            print(ingest_file(path))
        except ValueError as e:
            print(f"Skipped {filename}: {e}")


if __name__ == "__main__":
    ingest_directory(SOURCE_DIR)