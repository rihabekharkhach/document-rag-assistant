from app.rag.vector_store import retrieve_chunks
from app.rag.reranker import rerank

question = "Quelle précision obtient-on sur le gearbox et quelles classes sont confondues ?"

candidates = retrieve_chunks(question, top_k=20)
best = rerank(question, candidates, top_n=4)

for rank, chunk in enumerate(best, start=1):
    old_rank = candidates.index(chunk) + 1
    print(f"{rank} | was vector rank {old_rank} | {chunk[:90]!r}")