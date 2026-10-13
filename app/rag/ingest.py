# app/rag/ingest.py

"""
One-time (or on-demand) script to populate the vector database.
Run this whenever you add or update source documents:

    python -m app.rag.ingest
"""

import os
from app.rag.loaders import load_file
from app.rag.chunking import chunk_text
from app.rag.vector_store import ingest_documents, create_collection

SOURCE_DIR = "data/source_documents"  # put your PDFs/docx/images here


def ingest_directory(source_dir: str) -> None:
    """Load every supported file in a directory, chunk it, and store it in Qdrant."""
    create_collection()  # safe to call each run; skips if it already exists

    all_chunks: list[str] = []

    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)

        if not os.path.isfile(file_path):
            continue

        print(f"Loading: {filename}")
        try:
            raw_pages = load_file(file_path)
        except ValueError as e:
            print(f"  Skipped ({e})")
            continue

        full_text = "\n".join(raw_pages)
        chunks = chunk_text(full_text)
        print(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks produced — check your source directory and file types.")
        return

    print(f"\nIngesting {len(all_chunks)} chunks into Qdrant...")
    ingest_documents(all_chunks)
    print("Done.")


if __name__ == "__main__":
    ingest_directory(SOURCE_DIR)