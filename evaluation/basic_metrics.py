# evaluation/basic_metrics.py

"""Metrics computed directly from the saved outputs. No LLM call needed.
    python -m evaluation.basic_metrics
"""

import json
from pathlib import Path

records = json.loads(Path("evaluation/outputs_v2.json").read_text(encoding="utf-8"))

ok = [r for r in records if "error" not in r]
answerable = [r for r in ok if r["answerable"]]
unanswerable = [r for r in ok if not r["answerable"]]

latencies = sorted(r["latency_s"] for r in ok)
p90 = latencies[int(0.9 * (len(latencies) - 1))]

print(f"Questions: {len(records)} | errors: {len(records) - len(ok)}")
print(f"Abstention on unanswerable: {sum(not r['found'] for r in unanswerable)}/{len(unanswerable)}")
print(f"False abstention on answerable: {sum(not r['found'] for r in answerable)}/{len(answerable)}")
print(f"Answers found but with 0 verified citations: "
      f"{sum(r['found'] and r['n_citations'] == 0 for r in answerable)}/{len(answerable)}")
print(f"Latency: median {latencies[len(latencies) // 2]} s | p90 {p90} s")

print("\nAnswerable questions where the pipeline said 'not found':")
for r in answerable:
    if not r["found"]:
        print("  -", r["question"])