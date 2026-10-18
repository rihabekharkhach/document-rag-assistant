# evaluation/retrieval_metrics.py

"""
Retrieval-only comparison: vector search alone vs vector search + reranker.
No LLM judge: a chunk is a "hit" if it contains the gold snippet.
    python -m evaluation.retrieval_metrics
"""

import json
import unicodedata
from pathlib import Path

from app.rag.reranker import rerank
from app.rag.vector_store import retrieve_chunks

QUESTIONS_FILE = Path("evaluation/questions.jsonl")
CANDIDATES = 20  # what the vector search returns
K = 4            # what the generator receives


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if ch.isalnum())


def first_hit(chunks: list[str], snippet: str) -> int | None:
    """1-based rank of the first chunk containing the snippet, or None."""
    target = norm(snippet)
    for rank, chunk in enumerate(chunks, start=1):
        if target in norm(chunk):
            return rank
    return None


def mrr(rank: int | None) -> float:
    return 1 / rank if rank else 0.0


def main() -> None:
    lines = QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    items = [i for i in items if i.get("answerable") and i.get("gold_snippet")]

    ceiling = vec_hits = rr_hits = 0
    vec_mrr = rr_mrr = 0.0

    print(f"{'question':<48} | in top{CANDIDATES} | vector top{K} | reranked top{K}")
    for item in items:
        question, snippet = item["question"], item["gold_snippet"]
        candidates = retrieve_chunks(question, top_k=CANDIDATES)

        pos_all = first_hit(candidates, snippet)
        pos_vec = first_hit(candidates[:K], snippet)
        pos_rr = first_hit(rerank(question, candidates, top_n=K), snippet)

        ceiling += pos_all is not None
        vec_hits += pos_vec is not None
        rr_hits += pos_rr is not None
        vec_mrr += mrr(pos_vec)
        rr_mrr += mrr(pos_rr)

        print(f"{question[:48]:<48} | {str(pos_all):>9} | {str(pos_vec):>10} | {str(pos_rr):>13}")

    n = len(items)
    print(f"\nQuestions: {n}")
    print(f"Hit@{CANDIDATES} (retrieval ceiling): {ceiling}/{n}")
    print(f"Hit@{K} vector only : {vec_hits}/{n} | MRR {vec_mrr / n:.2f}")
    print(f"Hit@{K} + reranker  : {rr_hits}/{n} | MRR {rr_mrr / n:.2f}")


if __name__ == "__main__":
    main()