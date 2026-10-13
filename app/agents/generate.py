# app/agents/generate.py
from pydantic import BaseModel, Field

from app.agents.state import GraphState
from app.llm import llm


class Citation(BaseModel):
    chunk: int = Field(description="Number of the context chunk the quote comes from")
    quote: str = Field(description="Short excerpt copied EXACTLY, word for word, from that chunk")


class Answer(BaseModel):
    answer: str = Field(description="The answer to the question")
    found: bool = Field(description="False if the context does not contain the answer")
    citations: list[Citation] = Field(default_factory=list)


structured_llm = llm.with_structured_output(Answer)


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase, so PDF line breaks don't break the comparison."""
    return " ".join(text.split()).lower()


def verify_citations(citations: list[Citation], chunks: list[str]) -> list[dict]:
    """Keep only citations whose quote really appears in the chunk they claim."""
    verified = []
    for c in citations:
        index = c.chunk - 1
        if 0 <= index < len(chunks) and _normalize(c.quote) in _normalize(chunks[index]):
            verified.append({"chunk": c.chunk, "quote": c.quote})
    return verified


def generate_node(state: GraphState) -> dict:
    query = state["query"]
    chunks = state["chunks"]

    context = "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks, start=1))

    prompt = f"""You answer questions using ONLY the numbered context chunks below.

Question: {query}

Context:
{context}

Rules:
- Answer in the same language as the question, as briefly as the question allows.
- If the context does not contain the answer, set found=false and say so. Never use outside knowledge.
- Support your answer with citations: the chunk number and a short excerpt copied EXACTLY
  (word for word) from that chunk. Never paraphrase inside a quote.
- Quotes must stay in the original language of the chunk. Never translate a quote.
- Write the answer in the exact language of the question, even if the context is in another language."""

    result: Answer = structured_llm.invoke(prompt)

    verified = verify_citations(result.citations, chunks)
    print(f"[Citations] kept {len(verified)}/{len(result.citations)}")

    return {
        "answer": result.answer,
        "found": result.found,
        "citations": verified,
    }