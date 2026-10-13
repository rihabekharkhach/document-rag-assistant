# app/rag/embeddings.py

"""
Embedding model setup.

Loaded once here and imported wherever embeddings are needed
(ingestion and retrieval), so the model isn't reloaded into memory
multiple times across the project.
"""

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """Embed a query (e5 models expect the 'query: ' prefix)."""
    return embedder.encode("query: " + text, normalize_embeddings=True).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document chunks (e5 models expect the 'passage: ' prefix)."""
    return embedder.encode(["passage: " + t for t in texts], normalize_embeddings=True).tolist()