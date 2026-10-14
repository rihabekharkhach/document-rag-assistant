# app/models.py

"""
Pydantic schemas: the shape of what goes into and out of the API.
FastAPI uses them to validate requests automatically and to generate
the interactive documentation.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The user's question")
    thread_id: str | None = Field(
        default=None,
        description="Conversation id. Omit it to start a new conversation; "
                    "send the one you received to continue it.",
    )


class CitationOut(BaseModel):
    chunk: int
    quote: str


class QueryResponse(BaseModel):
    thread_id: str
    answer: str
    found: bool
    citations: list[CitationOut]
    standalone_query: str