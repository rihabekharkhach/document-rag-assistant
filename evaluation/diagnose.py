# evaluation/diagnose.py

"""For each question where the gold chunk is not ranked first after reranking, show the top 4.
    python -m evaluation.diagnose
"""

import json
import unicodedata
from pathlib import Path

from app.rag.reranker import rerank_indices
from app.rag.vector_store import retrieve_items


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if ch.isalnum())


lines = Path("evaluation/questions.jsonl").read_text(encoding="utf-8").splitlines()
for q in (json.loads(line) for line in lines if line.strip()):
    if not q.get("answerable") or not q.get("gold_snippet"):
        continue

    candidates = retrieve_items(q["question"], top_k=20)
    order = rerank_indices(q["question"], [c["text"] for c in candidates], top_n=4)
    top = [candidates[i] for i in order]
    gold = norm(q["gold_snippet"])
    hits = [gold in norm(c["text"]) for c in top]

    if not hits[0]:  # miss, or evidence not at rank 1
        print(f"\nQ: {q['question'][:80]} | gold: {q['gold_snippet']!r}")
        for rank, chunk in enumerate(top, start=1):
            mark = "✔" if hits[rank - 1] else " "
            print(f"  {rank}{mark} [{chunk.get('type')} p.{chunk.get('page')}] {chunk['text'][:90]!r}")