from app.agents.graph import research_graph

result = research_graph.invoke({
    "query": "Combien des conditions de fonctionnement existe dans le dataset ?",  # same question
    "research_notes": "",
    "critic_feedback": "",
    "is_sufficient": False,
    "retry_count": 0,
    "final_report": "",
})

print("Retries used:", result["retry_count"])
print("\n--- FINAL REPORT ---\n")
print(result["final_report"])