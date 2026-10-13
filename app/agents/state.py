# app/agents/state.py
from typing import TypedDict


class GraphState(TypedDict):
    query: str            # the user's question
    chunks: list[str]     # retrieved context chunks
    answer: str           # the final answer
    found: bool           # False if the documents don't contain the answer
    citations: list[dict] # verified citations: {"chunk": int, "quote": str}