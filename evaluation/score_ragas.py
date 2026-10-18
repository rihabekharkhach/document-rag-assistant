# evaluation/score_ragas.py

"""
Score the saved pipeline outputs with RAGAS, using Gemini as the judge.

    python -m evaluation.score_ragas 3     # smoke test on the first 3 questions
    python -m evaluation.score_ragas       # full run
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness

load_dotenv()

OUTPUTS_FILE = Path("evaluation/outputs_v2.json")
SCORES_FILE = Path("evaluation/scores_v2.csv")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")
JUDGE_RPM = float(os.getenv("JUDGE_RPM", "4"))


def build_judge() -> LangchainLLMWrapper:
    """Gemini as judge, throttled to stay under the free-tier requests-per-minute limit."""
    limiter = InMemoryRateLimiter(
        requests_per_second=JUDGE_RPM / 60, check_every_n_seconds=0.2, max_bucket_size=1
    )
    llm = ChatGoogleGenerativeAI(
        model=JUDGE_MODEL, temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries=4, rate_limiter=limiter,
    )
    return LangchainLLMWrapper(llm)


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    records = json.loads(OUTPUTS_FILE.read_text(encoding="utf-8"))
    kept = [r for r in records if r.get("answerable") and "error" not in r]
    if limit:
        kept = kept[:limit]

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["contexts"],
            response=r["answer"],
            reference=r["reference"],
        )
        for r in kept
    ]

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), FactualCorrectness()],
        llm=build_judge(),
        # Passed explicitly so RAGAS doesn't try to create a default OpenAI embedding model
        embeddings=LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
        ),
        run_config=RunConfig(max_workers=1, timeout=300),
        raise_exceptions=False,  # a failed judge call gives NaN instead of crashing the run
    )

    df = result.to_pandas()
    df.to_csv(SCORES_FILE, index=False, encoding="utf-8")

    scores = df.select_dtypes("number")
    print("\n--- Mean scores (NaN ignored) ---")
    print(scores.mean().round(3))
    print(f"\nNaN count per metric (failed judge calls):\n{scores.isna().sum()}")
    print("\n--- Per question ---")
    print(pd.concat([df["user_input"].str[:55], scores.round(2)], axis=1).to_string())


if __name__ == "__main__":
    main()