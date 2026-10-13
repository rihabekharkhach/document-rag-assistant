# test_graph.py
from app.agents.graph import research_graph

result = research_graph.invoke({"query": "Quel est le prix d’un bogie de train ?"})

print(result["answer"])
print("Found:", result["found"])
for c in result["citations"]:
    print(f'  [{c["chunk"]}] "{c["quote"]}"')