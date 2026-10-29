# evaluation/retrieval_metrics.py

"""
Dense vs hybrid retrieval, each with and without the reranker, per document.
    python -m evaluation.retrieval_metrics
Env: QUESTIONS_FILE, RETRIEVE_CANDIDATES (30), FINAL_CHUNKS (5), RERANK_PAUSE (2 seconds)
"""

import json
import os
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

from app.rag import reranker
from app.rag.reranker import rerank_indices
from app.rag.vector_store import retrieve_items

QUESTIONS_FILE = Path(os.getenv("QUESTIONS_FILE", "evaluation/questions.jsonl"))
CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "30"))
K = int(os.getenv("FINAL_CHUNKS", "5"))
PAUSE = float(os.getenv("RERANK_PAUSE", "2"))
SYSTEMS = ("dense", "dense+rerank", "hybrid", "hybrid+rerank")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if ch.isalnum())


def first_hit(texts: list[str], snippet: str) -> int | None:
    target = norm(snippet)
    for rank, text in enumerate(texts, start=1):
        if target in norm(text):
            return rank
    return None


def rerank_top(question: str, items: list[dict]) -> list[dict]:
    order = rerank_indices(question, [i["text"] for i in items], top_n=K)
    time.sleep(PAUSE)  # stay under the reranker's requests-per-minute limit
    return [items[i] for i in order]


def main() -> None:
    lines = QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    questions = [q for q in questions if q.get("answerable") and q.get("gold_snippet")]

    stats = defaultdict(lambda: {"n": 0, "right_doc": 0, **{s: [0, 0.0] for s in SYSTEMS}})

    print(f"rank of the evidence within the top {K} ('-' = missed)")
    print(f"{'question':<40} | " + " | ".join(f"{s:<13}" for s in SYSTEMS))
    for q in questions:
        question, snippet = q["question"], q["gold_snippet"]
        source = q.get("source", "(unspecified)")

        dense = retrieve_items(question, CANDIDATES, mode="dense")
        hybrid = retrieve_items(question, CANDIDATES, mode="hybrid")
        systems = {
            "dense": dense[:K],
            "dense+rerank": rerank_top(question, dense),
            "hybrid": hybrid[:K],
            "hybrid+rerank": rerank_top(question, hybrid),
        }
        ranks = {name: first_hit([i["text"] for i in items], snippet) for name, items in systems.items()}

        s = stats[source]
        s["n"] += 1
        top = systems["hybrid+rerank"]
        s["right_doc"] += bool(top) and top[0].get("source") == source
        for name in SYSTEMS:
            s[name][0] += ranks[name] is not None
            s[name][1] += 1 / ranks[name] if ranks[name] else 0.0

        print(f"{question[:40]:<40} | " + " | ".join(f"{str(ranks[n] or '-'):<13}" for n in SYSTEMS))

    print(f"\ncandidates per system: {CANDIDATES} | final chunks: {K}")
    for source, s in stats.items():
        n = s["n"]
        print(f"\n== {source} ({n} questions) | top chunk from this document: {s['right_doc']}/{n}")
        for name in SYSTEMS:
            print(f"   {name:<14} Hit@{K}: {s[name][0]}/{n} | MRR {s[name][1] / n:.2f}")

    if reranker.fallback_count:
        print(f"\n⚠️ {reranker.fallback_count} rerank call(s) fell back to vector order: the +rerank "
              f"columns are NOT valid. Raise RERANK_PAUSE (for example 5) and run again.")


if __name__ == "__main__":
    main()