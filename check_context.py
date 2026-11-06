from collections import Counter

from app.rag.vector_store import COLLECTION_NAME, client

counts, captioned, examples = Counter(), Counter(), []
offset = None
while True:
    points, offset = client.scroll(
        collection_name=COLLECTION_NAME, limit=500, offset=offset, with_payload=True, with_vectors=False
    )
    for point in points:
        payload = point.payload or {}
        if payload.get("type") != "text":
            continue
        source, text = payload.get("source"), payload["text"]
        counts[source] += 1
        if text.startswith(("[Table", "[Figure", "[Fig.")):
            captioned[source] += 1
        if "10 0.9685" in " ".join(text.split()):
            examples.append(text[:220])
    if offset is None:
        break

for source, n in counts.items():
    print(f"{source}: {n} text chunks, {captioned[source]} with a caption prefix")
print("\nchunk(s) containing the Table 2 epoch-10 row:")
for example in examples:
    print(repr(example))