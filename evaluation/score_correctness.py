# evaluation/score_correctness.py

"""
LLM-as-judge correctness: does the system's answer contain the key facts
of the reference answer? Two hand-made sanity cases run first, so you can
check that the judge really tells a right answer from a wrong one.

    python -m evaluation.score_correctness
"""

import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

OUTPUTS_FILE = Path("evaluation/outputs_v2.json")
SCORES_FILE = Path("evaluation/correctness_v2.csv")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")
JUDGE_RPM = float(os.getenv("JUDGE_RPM", "4"))

limiter = InMemoryRateLimiter(
    requests_per_second=JUDGE_RPM / 60, check_every_n_seconds=0.2, max_bucket_size=1
)
judge = ChatGoogleGenerativeAI(
    model=JUDGE_MODEL, temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=4, rate_limiter=limiter,
)

PROMPT = """You are grading the answer of a question-answering system against a reference answer.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Scoring rules:
- 1   : the system answer contains all key facts of the reference (same numbers, same items).
        Ignore wording, language, and extra correct detail.
- 0.5 : some key facts are present, others are missing.
- 0   : wrong, contradicts the reference, or misses the key facts.

Return ONLY a JSON object: {{"score": 0 or 0.5 or 1, "reason": "<one short sentence>"}}"""

SANITY = [
    ("Quelle précision est obtenue sur le gearbox ?", "95,4 %.", "La précision sur le gearbox est de 95,4 %.", 1),
    ("Quelle précision est obtenue sur le gearbox ?", "95,4 %.", "La précision sur le gearbox est de 62 %.", 0),
]


def grade(question: str, reference: str, answer: str) -> dict:
    text = judge.invoke(PROMPT.format(question=question, reference=reference, answer=answer)).content
    text = text if isinstance(text, str) else str(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0))
        return {"score": float(data["score"]), "reason": data.get("reason", "")}
    except (AttributeError, ValueError, KeyError):
        return {"score": float("nan"), "reason": f"unparseable: {text[:80]!r}"}


def main() -> None:
    print("--- Sanity check (expected scores: 1 then 0) ---")
    for q, ref, ans, expected in SANITY:
        print(f"expected {expected} -> {grade(q, ref, ans)}")

    records = json.loads(OUTPUTS_FILE.read_text(encoding="utf-8"))
    rows = []
    for r in records:
        if not r.get("answerable") or "error" in r:
            continue
        g = grade(r["question"], r["reference"], r["answer"])
        rows.append({"question": r["question"][:60], "modality": r.get("modality", "?"),"source": r.get("source", "?"), **g})
        print(f"{g['score']:>4} | {r['question'][:60]}")

    df = pd.DataFrame(rows)
    df.to_csv(SCORES_FILE, index=False, encoding="utf-8")

    print(f"\nMean correctness: {df['score'].mean():.2f}  (NaN: {df['score'].isna().sum()})")
    print(df.groupby(["source", "modality"])["score"].agg(["mean", "count"]).round(2))
    print("\nNot fully correct:")
    print(df[df["score"] < 1][["question", "score", "reason"]].to_string())


if __name__ == "__main__":
    main()