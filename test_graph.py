# test_graph.py
from app.agents.graph import research_graph

result = research_graph.invoke({"query": "Quel est le rôle de DANN dans l'architecture utilisée ?"})

print(result["answer"])
print("Found:", result["found"])
for c in result["citations"]:
    print(f'  [{c["chunk"]}] "{c["quote"]}"')