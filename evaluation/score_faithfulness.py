# evaluation/score_faithfulness.py

"""
Custom faithfulness judge: is every claim of the answer supported by the
retrieved context? Sanity cases run first.
    python -m evaluation.score_faithfulness
"""

import json
import re
from pathlib import Path

import pandas as pd

from evaluation.score_correctness import judge  # same Gemini judge + rate limiter

OUTPUTS_FILE = Path("evaluation/outputs_v2.json")
SCORES_FILE = Path("evaluation/faithfulness_v2.csv")

PROMPT = """You check whether an answer is supported by a context.

Context:
{context}

Answer: {answer}

Rules:
- The context may be in another language than the answer, and numbers may use a decimal
  comma or point (95,4 = 95.4). That is NOT a reason to mark a claim unsupported.
- List every factual claim of the answer and check each against the context only.
- 1   : every claim is supported by the context.
- 0.5 : some claims are supported, some are not found in the context.
- 0   : most claims are not supported or contradict the context.

Return ONLY a JSON object: {{"score": 0 or 0.5 or 1, "unsupported": ["claim", ...]}}"""

CTX = ("The model reaches an accuracy of 97.0% for the motor, 95.4% for the gearbox, "
       "99.8% for the left axle box, and 99.97% for the right axle box.")

SANITY = [
    ("French answer, comma decimal (expect 1)", CTX, "La précision sur le gearbox est de 95,4 %."),
    ("English answer (expect 1)", CTX, "The gearbox accuracy is 95.4%."),
    ("Invented claims (expect <= 0.5)", CTX,
     "La précision sur le gearbox est de 95,4 %, obtenue avec un réseau de 12 couches entraîné pendant 3 jours."),
]


def grade(context: str, answer: str) -> dict:
    text = judge.invoke(PROMPT.format(context=context, answer=answer)).content
    text = text if isinstance(text, str) else str(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0))
        return {"score": float(data["score"]), "unsupported": data.get("unsupported", [])}
    except (AttributeError, ValueError, KeyError):
        return {"score": float("nan"), "unsupported": [f"unparseable: {text[:80]!r}"]}


def main() -> None:
    print("--- Sanity check ---")
    for label, ctx, ans in SANITY:
        print(f"{label}: {grade(ctx, ans)}")

    records = json.loads(OUTPUTS_FILE.read_text(encoding="utf-8"))
    rows = []
    for r in records:
        if not r.get("answerable") or "error" in r or not r.get("found"):
            continue
        g = grade("\n\n".join(r["contexts"]), r["answer"])
        rows.append({"question": r["question"][:60], **g})
        print(f"{g['score']:>4} | {r['question'][:60]}")

    df = pd.DataFrame(rows)
    df.to_csv(SCORES_FILE, index=False, encoding="utf-8")
    print(f"\nMean faithfulness: {df['score'].mean():.2f}  (NaN: {df['score'].isna().sum()})")
    print("\nNot fully faithful:")
    print(df[df["score"] < 1].to_string())


if __name__ == "__main__":
    main()