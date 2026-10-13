# app/rag/chunking.py

"""
Text chunking: splits raw extracted text into smaller overlapping
pieces before embedding.

Why chunk at all: embedding models work best on short, focused
passages — a whole document embedded as one vector loses too much
specificity for accurate retrieval. Overlap keeps context from being
cut off mid-idea at chunk boundaries.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800       # characters per chunk — small enough for precise retrieval, large enough to keep context
CHUNK_OVERLAP = 100    # characters shared between consecutive chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def chunk_text(text: str) -> list[str]:
    """Split a single block of text into overlapping chunks."""
    return splitter.split_text(text)