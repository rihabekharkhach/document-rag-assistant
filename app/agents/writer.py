# app/agents/writer.py

"""
Writer node: takes the final, critic-approved research and produces
a structured, readable report. This is the last node in the graph —
after this runs, execution ends.
"""

from langchain_ollama import ChatOllama

from app.agents.state import GraphState

llm = ChatOllama(model="llama3.1:8b")


def writer_node(state: GraphState) -> dict:
    query = state["query"]
    research_notes = state.get("research_notes", "")

    prompt = f"""You are writing a final report to answer this question:

Question: {query}

Approved research findings:
{research_notes}

Answer in the same language as the question, using only the research findings above.
Match the length to the question: a simple factual question gets a direct answer in
1-3 sentences plus one line of supporting evidence. Use sections only for complex questions.
State numbers exactly as given in the findings; do not hedge ("at least") unless the
findings themselves are uncertain."""

    response = llm.invoke(prompt)

    return {"final_report": response.content}