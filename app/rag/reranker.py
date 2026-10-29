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
import time
from dotenv import load_dotenv

load_dotenv()

JINA_URL = "https://api.jina.ai/v1/rerank"
JINA_MODEL = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v3.5")


fallback_count = 0  # how many times the reranker could not be used (evaluation scripts check it)


def _wait_seconds(response, attempt: int) -> float:
    try:
        return min(float(response.headers.get("Retry-After", "")), 30)
    except ValueError:
        return 2 * (attempt + 1)


def rerank_indices(query: str, chunks: list[str], top_n: int = 4) -> list[int]:
    """Indices of the top_n chunks, most relevant first. Falls back to the original order."""
    global fallback_count
    fallback = list(range(min(top_n, len(chunks))))
    if len(chunks) <= top_n:
        return fallback
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        fallback_count += 1
        print("[Rerank] JINA_API_KEY not set: using vector-search order")
        return fallback

    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(
                JINA_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": JINA_MODEL, "query": query, "documents": chunks, "top_n": top_n},
                timeout=30,
            )
            if response.status_code in (429, 500, 502, 503):  # rate limit or temporary server error
                last_error = f"HTTP {response.status_code}"
                time.sleep(_wait_seconds(response, attempt))
                continue
            response.raise_for_status()
            results = sorted(response.json()["results"], key=lambda r: r["relevance_score"], reverse=True)
            return [r["index"] for r in results]
        except (requests.RequestException, KeyError, ValueError) as e:
            last_error = e
            time.sleep(2 * (attempt + 1))

    fallback_count += 1
    print(f"[Rerank] failed after 3 attempts ({last_error}): using vector-search order")
    return fallback


def rerank(query: str, chunks: list[str], top_n: int = 4) -> list[str]:
    """Same as before, returning the chunk texts. Kept so the evaluation scripts keep working."""
    return [chunks[i] for i in rerank_indices(query, chunks, top_n)]