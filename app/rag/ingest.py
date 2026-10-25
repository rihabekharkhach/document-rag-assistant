# app/rag/ingest.py

"""
Populate the vector database from data/source_documents:
text chunks, plus one element per figure (caption + OCR text) for PDFs.

    python -m app.rag.ingest
"""

import os

from app.rag.chunking import chunk_text
from app.rag.figures import Figure, extract_figures
from app.rag.loaders import load_file
from app.rag.ocr import ocr_image
from app.rag.vector_store import create_collection, ingest_documents

SOURCE_DIR = "data/source_documents"
MAX_FIGURE_CHARS = 2000  # keeps one figure within the embedding model's input size


def figure_item(fig: Figure) -> dict:
    """One searchable element per figure: caption first (anchors the meaning), then the OCR text."""
    ocr_text = ocr_image(fig.image_path)[:MAX_FIGURE_CHARS]
    caption = fig.caption or "Figure (caption not found)"
    return {
        "text": f"[Figure, page {fig.page}] {caption}\nText detected in the figure:\n{ocr_text}",
        "type": "figure",
        "source": fig.source,
        "page": fig.page,
        "image_path": fig.image_path,
    }


def ingest_directory(source_dir: str) -> None:
    create_collection()
    items: list[dict] = []

    for filename in os.listdir(source_dir):
        path = os.path.join(source_dir, filename)
        if not os.path.isfile(path):
            continue

        print(f"Loading: {filename}")
        try:
            raw_pages = load_file(path)
        except ValueError as e:
            print(f"  Skipped ({e})")
            continue

        text_items = [
            {"text": chunk, "type": "text", "source": filename}
            for chunk in chunk_text("\n".join(raw_pages))
        ]
        print(f"  → {len(text_items)} text chunks")
        items.extend(text_items)

        if filename.lower().endswith(".pdf"):
            figures = extract_figures(path)
            print(f"  → {len(figures)} figures (OCR)")
            items.extend(figure_item(f) for f in figures)

    if not items:
        print("Nothing to ingest: check your source directory.")
        return

    print(f"\nIngesting {len(items)} elements into Qdrant...")
    ingest_documents(items)
    print("Done.")


if __name__ == "__main__":
    ingest_directory(SOURCE_DIR)