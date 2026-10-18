# evaluation/collect.py

"""
Run the RAG pipeline on every question of the golden set and save
question, contexts, answer and timing, ready for RAGAS scoring.

    python -m evaluation.collect
"""

import json
import time
import uuid
from pathlib import Path

from app.agents.graph import research_graph

QUESTIONS_FILE = Path("evaluation/questions.jsonl")
OUTPUT_FILE = Path("evaluation/outputs_v2.json")


def load_questions() -> list[dict]:
    lines = QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> None:
    records = []
    for item in load_questions():
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        start = time.time()
        try:
            result = research_graph.invoke({"query": item["question"]}, config=config)
        except Exception as e:
            print(f"[ERROR] {type(e).__name__} | {item['question'][:70]}")
            records.append({**item, "error": type(e).__name__})
            OUTPUT_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            continue
        latency = round(time.time() - start, 1)

        records.append({
            "question": item["question"],
            "reference": item["reference"],
            "answerable": item["answerable"],
            "answer": result["answer"],
            "found": result["found"],
            "contexts": result["chunks"],
            "n_citations": len(result["citations"]),
            "latency_s": latency,
        })
        print(f"[{latency:>5}s] found={result['found']!s:<5} | {item['question'][:70]}")
        OUTPUT_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()