import statistics
from collections import Counter

from app.rag.vector_store import COLLECTION_NAME, client

lengths, per_page = [], Counter()
offset = None
while True:
    points, offset = client.scroll(
        collection_name=COLLECTION_NAME, limit=500, offset=offset, with_payload=True, with_vectors=False
    )
    for point in points:
        payload = point.payload or {}
        if payload.get("type") == "text":
            lengths.append(len(payload["text"]))
            per_page[(payload.get("source"), payload.get("page"))] += 1
    if offset is None:
        break

print(f"text chunks: {len(lengths)} | min {min(lengths)} | median {int(statistics.median(lengths))} | max {max(lengths)} characters")
print("chunks per page:", dict(sorted(per_page.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0))))