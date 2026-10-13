# app/rag/vector_store.py

"""
Qdrant vector store: owns the client connection, collection setup,
document ingestion (embed + upsert), and retrieval (embed query +
similarity search).

This is the only file in the project that talks to Qdrant directly —
other files (ingest.py, researcher.py) call ingest_documents() /
retrieve() without knowing how the storage works underneath.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from app.rag.embeddings import embed_text, embed_texts, EMBEDDING_DIM

COLLECTION_NAME = "docs"

# Local Qdrant running in Docker (docker-compose) — swap to :memory: only for quick throwaway tests
client = QdrantClient(host="localhost", port=6333)


def create_collection() -> None:
    """Create the Qdrant collection if it doesn't already exist. Safe to call on every run."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


import uuid

def ingest_documents(chunks: list[str]) -> None:
    """Embed chunks and upsert them. IDs come from the text, so re-ingesting never duplicates."""
    vectors = embed_texts(chunks)

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk)),
            vector=vectors[i],
            payload={"text": chunk},
        )
        for i, chunk in enumerate(chunks)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)


def retrieve(query: str, top_k: int = 5) -> str:
    """Embed a query and return the top-k most similar chunks, joined as one string."""
    query_vector = embed_text(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )

    return "\n\n".join(hit.payload["text"] for hit in results.points)