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
from bisect import bisect_right
import re


CHUNK_SIZE = 800       # characters per chunk — small enough for precise retrieval, large enough to keep context
CHUNK_OVERLAP = 100    # characters shared between consecutive chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

def chunk_text(text: str) -> list[str]:
    """Split a single block of text into overlapping chunks."""
    return splitter.split_text(text)

position_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
CAPTION = re.compile(r"^(?:Table|Figure|Fig\.)\s+\d+[^\n]*", re.MULTILINE | re.IGNORECASE)


def chunk_pages(pages: list[str]) -> list[tuple[str, int]]:
    """Split the whole document, then tag each chunk with the page where it starts."""
    page_starts, offset = [], 0
    for page_text in pages:
        page_starts.append(offset)
        offset += len(page_text) + 1  # +1 for the "\n" separator
    full_text = "\n".join(pages)

    chunks = position_splitter.create_documents([full_text])
    return [
        (doc.page_content, bisect_right(page_starts, doc.metadata["start_index"]))
        for doc in chunks
    ]


def chunk_page_with_context(page_text: str) -> list[str]:
    """Split one page; prefix each chunk with the nearest caption above it (Table N / Figure N)."""
    captions = [(m.start(), m.group(0).strip()[:140]) for m in CAPTION.finditer(page_text)]
    result = []
    for doc in position_splitter.create_documents([page_text]):
        above = [c for pos, c in captions if pos <= doc.metadata["start_index"]]
        text = doc.page_content
        if above and above[-1] not in text:
            text = f"[{above[-1]}]\n{text}"
        result.append(text)
    return result