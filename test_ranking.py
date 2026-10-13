from app.rag.embeddings import embed_text
from app.rag.vector_store import client, COLLECTION_NAME


def show(question: str, k: int = 10) -> None:
    print(f"\nQ: {question}")
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embed_text(question),
        limit=k,
        with_payload=True,
    )
    for rank, hit in enumerate(res.points, 1):
        text = hit.payload["text"]
        flag = "  <-- ANSWER CHUNK" if "_parent" in text else ""
        print(f"{rank:>2} | score={hit.score:.3f} | {text[:60]!r}{flag}")


show('Quelle est la différence entre target="_blank" et "_self" dans une balise a ?')
show("Comment ouvrir un lien dans un nouvel onglet en HTML ?")