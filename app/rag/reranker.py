# app/rag/reranker.py

"""
Reranker: re-scores retrieved chunks against the question using a
cross-encoder served by an API, and keeps only the best ones.

Why: vector search compares two separate embeddings (fast but approximate).
A cross-encoder reads the question AND each chunk together, which is
much more precise, but too slow to run on the whole database. So the
standard pattern is: vector search for ~20 candidates, rerank to ~4.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

JINA_URL = "https://api.jina.ai/v1/rerank"
JINA_MODEL = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v3.5")


def rerank(query: str, chunks: list[str], top_n: int = 4) -> list[str]:
    """Return the top_n chunks, most relevant first.

    If the API key is missing or the call fails, fall back to the original
    vector-search order, so a reranker outage never breaks the pipeline.
    """
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        print("[Rerank] JINA_API_KEY not set: using vector-search order")
        return chunks[:top_n]
    if len(chunks) <= top_n:
        return chunks

    try:
        response = requests.post(
            JINA_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": JINA_MODEL, "query": query, "documents": chunks, "top_n": top_n},
            timeout=30,
        )
        response.raise_for_status()
        results = response.json()["results"]
        results = sorted(results, key=lambda r: r["relevance_score"], reverse=True)
        return [chunks[r["index"]] for r in results]
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"[Rerank] failed ({e}): using vector-search order")
        return chunks[:top_n]