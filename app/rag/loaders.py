# app/rag/loaders.py

"""
Document loaders: convert files of various formats (PDF, DOCX, images)
into raw text. Text-based files use direct extraction; images and
scanned PDFs use a local vision-language model (VLM) for OCR.
"""

import base64
from io import BytesIO

from PIL import Image
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# --- VLM setup (used for images and scanned/OCR PDFs) ---

vlm = ChatOllama(model="qwen2.5vl:7b")

MIN_TEXT_LENGTH = 20  # if extracted text is shorter than this, assume the PDF is scanned


def image_to_base64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string for the VLM."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_text_from_image(image: Image.Image) -> str:
    """Send one image to the VLM and return the extracted text."""
    image_b64 = image_to_base64(image)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Extract all text from this document image, preserving structure and reading order.",
            },
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"},
        ]
    )
    response = vlm.invoke([message])
    return response.content


# --- Text-based extraction (fast path, no VLM needed) ---


def load_pdf_text(file_path: str) -> list[str]:
    """Extract text directly from a text-based PDF (not scanned)."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return [doc.page_content for doc in documents]


def load_docx_text(file_path: str) -> list[str]:
    """Extract text directly from a Word document."""
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    return [doc.page_content for doc in documents]

# add this new function, right after load_docx_text() and before the VLM section

def load_pdf(file_path: str) -> list[str]:
    """Try direct text extraction first; fall back to VLM if the PDF turns out to be scanned."""
    pages = load_pdf_text(file_path)

    total_length = sum(len(p.strip()) for p in pages)
    if total_length < MIN_TEXT_LENGTH:
        print(f"'{file_path}' looks scanned (almost no extractable text) — falling back to VLM.")
        return load_pdf_vlm(file_path)

    return pages

# --- VLM-based extraction (for images and scanned PDFs) ---


def load_image_vlm(file_path: str) -> list[str]:
    """Extract text from a single image file using the VLM."""
    image = Image.open(file_path)
    return [extract_text_from_image(image)]


def load_pdf_vlm(file_path: str) -> list[str]:
    """Extract text from a scanned PDF by rendering each page as an
    image and running the VLM on it."""
    pages = convert_from_path(file_path)
    return [extract_text_from_image(page) for page in pages]


# --- Public entry point ---


# replace the old load_file() with this — no more 'scanned' parameter

def load_file(file_path: str) -> list[str]:
    """
    Load a file and return its content as a list of text chunks
    (one per page/section, before splitting). Automatically detects
    scanned PDFs and routes them to the VLM.
    """
    if file_path.endswith(".pdf"):
        return load_pdf(file_path)
    elif file_path.endswith(".docx"):
        return load_docx_text(file_path)
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        return load_image_vlm(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    