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
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchAny, MatchValue, PointStruct, VectorParams,
)


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

def ingest_documents(items: list[dict]) -> None:
    """items: [{"text": str, "type": "text" | "figure", "source": str,
                "page": int (figures), "image_path": str (figures)}]"""
    vectors = embed_texts([item["text"] for item in items])

    points = [
        PointStruct(
            # source + type in the ID: the same text in two documents stays two points
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{item.get('source', '')}|{item.get('type', 'text')}|{item['text']}")),
            vector=vectors[i],
            payload=item,  # contains "text", so retrieve_chunks keeps working unchanged
        )
        for i, item in enumerate(items)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def _source_filter(sources: list[str] | None) -> Filter | None:
    """Restrict a search to some documents. None or empty = search everything."""
    if not sources:
        return None
    return Filter(must=[FieldCondition(key="source", match=MatchAny(any=sources))])


def retrieve_chunks(query: str, top_k: int = 5, sources: list[str] | None = None) -> list[str]:
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embed_text(query),
        limit=top_k,
        query_filter=_source_filter(sources),
    )
    return [hit.payload["text"] for hit in results.points]


def delete_source(source: str) -> None:
    """Remove every point that belongs to one document."""
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
    )


def list_sources() -> list[dict]:
    """Aggregate the collection by document: name, number of text chunks, number of figures."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        return []

    stats: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME, limit=500, offset=offset,
            with_payload=["source", "type"], with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            name = payload.get("source", "(inconnu)")
            entry = stats.setdefault(name, {"name": name, "text_chunks": 0, "figures": 0})
            entry["figures" if payload.get("type") == "figure" else "text_chunks"] += 1
        if offset is None:
            break
    return sorted(stats.values(), key=lambda d: d["name"])


def retrieve(query: str, top_k: int = 3) -> str:
    """Same as before, kept so your test scripts still work."""
    return "\n\n".join(retrieve_chunks(query, top_k))