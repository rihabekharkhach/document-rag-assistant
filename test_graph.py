from app.agents.graph import research_graph

config = {"configurable": {"thread_id": "test-1"}}  # one thread = one conversation

history = []

def ask(question: str) -> None:
    result = research_graph.invoke({"query": question, "chat_history": history})
    history.extend([{"role": "user", "content": question},
                    {"role": "assistant", "content": result["answer"]}])
    print(f"\nQ: {question}\n   (reformulée : {result['standalone_query']})\nA: {result['answer']}")


ask("Quelles sont les conditions de fonctionnement utilisées pour le test ?")
ask("Et pour la validation ?")
ask("Combien y en a-t-il au total ?")