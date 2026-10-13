# app/agents/researcher.py

"""
Researcher node: given the current graph state, retrieves relevant
context from the vector store and asks the LLM to produce findings.
On retry (when the Critic flagged something missing), it searches
using the critic's feedback instead of the original query, and
appends new findings to what's already been gathered.
"""
from app.llm import llm

from app.agents.state import GraphState
from app.rag.vector_store import retrieve

#llm = ChatOllama(model="llama3.1:8b")


def researcher_node(state: GraphState) -> dict:
    query = state["query"]
    feedback = state.get("critic_feedback", "")

    # On retry, search for what was specifically missing; otherwise search the original question
    search_term = feedback if feedback else query
    retrieved_context = retrieve(search_term)

    prompt = f"""Research question: {query}

Relevant context from documents:
{retrieved_context}

{"The critic said this was missing from the research so far: " + feedback if feedback else ""}

Provide detailed findings to help answer this question, grounded in the context above.
If the context doesn't fully cover the question, say clearly what's missing."""

    response = llm.invoke(prompt)

    # Append, don't overwrite — research accumulates across retries
    existing_notes = state.get("research_notes", "")
    updated_notes = (existing_notes + "\n\n" + response.content).strip()

    return {"research_notes": updated_notes}