# app/agents/state.py

"""
Shared state schema passed between every node in the graph.
Each node reads from this and returns only the fields it changes;
LangGraph merges updates into the full state automatically.
"""

from typing import TypedDict


class GraphState(TypedDict):
    query: str              # the original user question — never changes
    research_notes: str     # accumulated findings from the Researcher, grows on retries
    critic_feedback: str    # what's missing, set by the Critic; empty string when sufficient
    is_sufficient: bool     # decision flag the conditional edge reads to route researcher vs. writer
    retry_count: int        # safety counter, capped in critic.py to avoid infinite loops
    final_report: str       # the Writer's output — populated only at the end