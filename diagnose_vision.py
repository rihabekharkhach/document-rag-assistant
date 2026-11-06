import sys

from app.agents.graph import research_graph

question = sys.argv[1]
result = research_graph.invoke({"query": question, "chat_history": [], "sources": None})
print("\nanswer_mode:", result.get("answer_mode", "text"), "| found:", result["found"])
print("answer:", result["answer"])
print("top chunks after reranking:")
for rank, item in enumerate(result["items"], start=1):
    print(f"  {rank}. {item.get('type')} | {item.get('source')} | p.{item.get('page')}")