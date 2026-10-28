# app/agents/state.py
from typing import TypedDict


class GraphState(TypedDict):
    query: str               # the raw question as the user typed it
    chat_history: list[dict] # [{"role": "user" | "assistant", "content": str}]
    standalone_query: str    # the question rewritten to stand on its own
    chunks: list[str]
    answer: str
    found: bool
    citations: list[dict]
    sources: list[str] | None   # documents to search; None = all
    items: list[dict]         # retrieved elements with metadata (same order as chunks)
    sources_used: list[dict]  # metadata of the chunks given to the generator