# app/models.py

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "Quelles sont les conditions de fonctionnement utilisées pour le test ?"}]
        }
    }

    question: str = Field(..., min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, description="Conversation id. Omit to start a new one.")
    sources: list[str] | None = Field(default=None, description="Document names to search. Omit for all.")


class SourceOut(BaseModel):          # must come before QueryResponse and MessageOut
    source: str | None = None
    page: int | None = None
    type: str = "text"
    image_path: str | None = None


class CitationOut(BaseModel):
    chunk: int
    quote: str


class QueryResponse(BaseModel):
    thread_id: str
    answer: str
    found: bool
    citations: list[CitationOut]
    standalone_query: str
    sources: list[SourceOut] = []
    answer_mode: str = "text"


class IngestReportOut(BaseModel):
    name: str
    status: str
    pages: int = 0
    characters: int = 0
    text_chunks: int = 0
    figures: int = 0
    seconds: float = 0.0
    message: str = ""


class DocumentOut(BaseModel):
    name: str
    text_chunks: int
    figures: int


class ConversationOut(BaseModel):
    id: str
    title: str
    updated_at: str


class MessageOut(BaseModel):
    role: str
    content: str
    found: bool | None = None
    citations: list[CitationOut] = []
    standalone_query: str | None = None
    sources: list[SourceOut] = []
    created_at: str
    answer_mode: str = "text"