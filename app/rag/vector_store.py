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
from app.rag.embeddings import embed_text, embed_texts, EMBEDDING_DIM
import os
from qdrant_client.models import (
    Distance, FieldCondition, Filter, Fusion, FusionQuery, MatchAny, MatchValue,
    Modifier, PointStruct, Prefetch, SparseVectorParams, VectorParams,
)

from app.rag.sparse import sparse_document, sparse_query
import uuid


COLLECTION_NAME = "docs"

# Local Qdrant running in Docker (docker-compose) — swap to :memory: only for quick throwaway tests
client = QdrantClient(host="localhost", port=6333)

def create_collection() -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )


def ingest_documents(items: list[dict]) -> None:
    dense_vectors = embed_texts([item["text"] for item in items])
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{item.get('source', '')}|{item.get('type', 'text')}|{item['text']}")),
            vector={"dense": dense_vectors[i], "sparse": sparse_document(item["text"])},
            payload=item,
        )
        for i, item in enumerate(items)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def retrieve_items(query: str, top_k: int = 5, sources: list[str] | None = None,
                   mode: str | None = None) -> list[dict]:
    """mode: 'dense', 'sparse' or 'hybrid' (default, or set RETRIEVAL_MODE in .env)."""
    mode = mode or os.getenv("RETRIEVAL_MODE", "dense")
    flt = _source_filter(sources)
    sparse = sparse_query(query)

    if mode == "sparse":
        if not sparse.indices:
            return []
        response = client.query_points(collection_name=COLLECTION_NAME, query=sparse,
                                       using="sparse", limit=top_k, query_filter=flt)
    elif mode == "dense":
        response = client.query_points(collection_name=COLLECTION_NAME, query=embed_text(query),
                                       using="dense", limit=top_k, query_filter=flt)
    else:
        legs = [Prefetch(query=embed_text(query), using="dense", limit=top_k, filter=flt)]
        if sparse.indices:  # a query with no usable tokens falls back to dense only
            legs.append(Prefetch(query=sparse, using="sparse", limit=top_k, filter=flt))
        response = client.query_points(collection_name=COLLECTION_NAME, prefetch=legs,
                                       query=FusionQuery(fusion=Fusion.RRF), limit=top_k)
    return [hit.payload for hit in response.points]


def retrieve_chunks(query: str, top_k: int = 5, sources: list[str] | None = None,
                    mode: str | None = None) -> list[str]:
    return [item["text"] for item in retrieve_items(query, top_k, sources, mode)]


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


def retrieve_chunks(query: str, top_k: int = 5, sources: list[str] | None = None) -> list[str]:
    return [item["text"] for item in retrieve_items(query, top_k, sources)]