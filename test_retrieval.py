from app.rag.vector_store import retrieve

question = "Combien des conditions de fonctionnement existe dans le dataset ?"  # adapt this
print(retrieve(question, top_k=3))