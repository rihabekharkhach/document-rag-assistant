from app.rag.vector_store import COLLECTION_NAME, client

offset = None
while True:
    points, offset = client.scroll(
        collection_name=COLLECTION_NAME, limit=200, offset=offset, with_payload=True, with_vectors=False
    )
    for point in points:
        payload = point.payload or {}
        if payload.get("type") == "figure" and payload.get("page") == 19:
            print(payload.get("source"), "| image:", payload.get("image_path"))
            print("   nearby:", repr(payload.get("nearby", "<<missing>>"))[:300])
    if offset is None:
        break