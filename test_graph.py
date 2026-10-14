from app.agents.graph import research_graph

config = {"configurable": {"thread_id": "test-1"}}  # one thread = one conversation


def ask(question: str) -> None:
    result = research_graph.invoke({"query": question}, config=config)
    print(f"\nQ: {question}")
    print(f"   (reformulée : {result['standalone_query']})")
    print(f"A: {result['answer']}  [found={result['found']}, citations={len(result['citations'])}]")


ask("Quelles sont les conditions de fonctionnement utilisées pour le test ?")
ask("Et pour la validation ?")
ask("Combien y en a-t-il au total ?")