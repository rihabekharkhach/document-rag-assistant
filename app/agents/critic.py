# app/agents/critic.py

"""
Critic node: reviews the research gathered so far and decides
whether it's sufficient to write a good final report, or whether
the Researcher needs to go back and dig deeper.

This node doesn't just approve/reject — it also increments the
retry counter (safety limit) and produces specific feedback the
Researcher can act on.
"""

import re

from langchain_ollama import ChatOllama

from app.agents.state import GraphState

llm = ChatOllama(model="llama3.1:8b")

MAX_RETRIES = 3


def critic_node(state: GraphState) -> dict:
    query = state["query"]
    research_notes = state.get("research_notes", "")
    retry_count = state.get("retry_count", 0)

    prompt = f"""You are reviewing research notes gathered to answer a question.

Question: {query}

Research notes:
{research_notes}

Decide ONLY whether the notes contain enough information to answer THIS question.
Do not ask for extra details the question does not require.
If the answer can be stated from the notes, it is SUFFICIENT.

Respond in exactly this format:
DECISION: SUFFICIENT or INSUFFICIENT
FEEDBACK: <if insufficient, name the specific missing fact needed to answer the question; otherwise "none">"""

    response = llm.invoke(prompt).content

    # Parse the LLM's structured response
    # is_sufficient = "DECISION: SUFFICIENT" in response.upper()

    match = re.search(r"DECISION\W*\s*(INSUFFICIENT|SUFFICIENT)", response, re.IGNORECASE)
    is_sufficient = bool(match) and match.group(1).upper() == "SUFFICIENT"

    feedback_line = ""
    for line in response.splitlines():
        if line.strip().upper().startswith("FEEDBACK:"):
            feedback_line = line.split(":", 1)[1].strip()

    # Force-approve if we've already retried too many times,
    # so the graph never loops forever on a stubborn Critic

    print(f"[Critic] pass {retry_count + 1} | sufficient={is_sufficient} | feedback={feedback_line!r}")
    
    if retry_count >= MAX_RETRIES:
        is_sufficient = True

    return {
        "is_sufficient": is_sufficient,
        "critic_feedback": "" if is_sufficient else feedback_line,
        "retry_count": retry_count + 1,
    }