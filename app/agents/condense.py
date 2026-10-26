# app/agents/condense.py
from app.agents.state import GraphState
from app.llm import llm
import unicodedata

MAX_HISTORY_MESSAGES = 6  # last 3 exchanges: enough context, small prompt

def _same(a: str, b: str) -> bool:
    def keep(t: str) -> str:
        return "".join(ch for ch in unicodedata.normalize("NFKC", t).lower() if ch.isalnum())
    return keep(a) == keep(b)


def condense_node(state: GraphState) -> dict:
    query = state["query"]
    history = state.get("chat_history", [])

    # First question: nothing to resolve, so skip the LLM call entirely
    if not history:
        return {"standalone_query": query}

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-MAX_HISTORY_MESSAGES:])

    prompt = f"""Rewrite the follow-up question as a standalone question that can be
understood without the conversation.
- Keep the language of the follow-up question.
- If it is already standalone, return it unchanged.
- Do NOT answer it. Output only the rewritten question.

Conversation:
{transcript}

Follow-up question: {query}
Standalone question:"""

    answer = llm.invoke(prompt).content.strip().strip('"«»“”')
    if not answer or answer.upper().rstrip(".") == "SAME" or _same(answer, query):
        return {"standalone_query": query}
    return {"standalone_query": answer}