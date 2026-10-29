# app/agents/generate.py
from pydantic import BaseModel, Field

from app.agents.state import GraphState
from app.llm import llm

import json
import re
import unicodedata


class Citation(BaseModel):
    chunk: int = Field(description="Number of the context chunk the quote comes from")
    quote: str = Field(description="Short excerpt copied EXACTLY, word for word, from that chunk")


class Answer(BaseModel):
    answer: str = Field(description="The answer to the question")
    found: bool = Field(description="False if the context does not contain the answer")
    citations: list[Citation] = Field(default_factory=list)


structured_llm = llm.with_structured_output(Answer)


def _normalize(text: str) -> str:
    """Keep only lowercase letters and digits, so line breaks, hyphenation,
    quote styles and punctuation from the PDF can't cause false rejections."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if ch.isalnum())


MIN_QUOTE_CHARS = 10
MIN_NUMERIC_QUOTE_CHARS = 3  # numbers like "0.31" are legitimate short quotes


def _is_supported(quote: str, chunk_norm: str) -> bool:
    """A quote may contain '...': each segment must appear in the chunk."""
    segments = [s for s in re.split(r"\.\.\.|…", quote) if s.strip()]
    if not segments:
        return False
    for segment in segments:
        s = _normalize(segment)
        min_len = MIN_NUMERIC_QUOTE_CHARS if any(ch.isdigit() for ch in s) else MIN_QUOTE_CHARS
        if len(s) < min_len or s not in chunk_norm:
            return False
    return True


def verify_citations(citations: list[Citation], chunks: list[str]) -> list[dict]:
    verified = []
    for c in citations:
        index = c.chunk - 1
        if 0 <= index < len(chunks) and _is_supported(c.quote, _normalize(chunks[index])):
            verified.append({"chunk": c.chunk, "quote": c.quote})
        else:
            print(f"[Citations] rejected: chunk={c.chunk} quote={c.quote!r}")
    return verified

def _parse_json_answer(text: str) -> Answer:
    """Extract the first {...} block from plain model output and validate it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model output")
    return Answer.model_validate(json.loads(match.group(0)))


def _invoke_structured(prompt: str) -> Answer:
    """Try native structured output; on failure, fall back to plain JSON parsing."""
    try:
        return structured_llm.invoke(prompt)
    except Exception as e:
        print(f"[Generate] structured output failed ({type(e).__name__}): falling back to plain JSON")

    text = llm.invoke(
        prompt
        + "\n\nReturn ONLY a JSON object with the keys: answer (string), found (boolean), "
          "citations (list of objects with chunk (int) and quote (string)). No other text."
    ).content
    return _parse_json_answer(text)

def generate_node(state: GraphState) -> dict:
    question = state["standalone_query"] 
    chunks = state["chunks"]

    context = "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks, start=1))

    prompt = f"""You answer questions using ONLY the numbered context chunks below.

Question: {question}

Context:
{context}

Rules:
- Answer in the same language as the question, as briefly as the question allows.
- If the context does not contain the answer, set found=false and say so. Never use outside knowledge.
- Support your answer with citations: the chunk number and a short excerpt copied EXACTLY
  (word for word) from that chunk. Never paraphrase inside a quote.
- Quotes must stay in the original language of the chunk. Never translate a quote.
- Write the answer in the exact language of the question, even if the context is in another language."""

    result: Answer = _invoke_structured(prompt)
    answer = result.answer.strip() or "Je n'ai pas trouvé cette information dans les documents."

    verified = verify_citations(result.citations, chunks)
    print(f"[Citations] kept {len(verified)}/{len(result.citations)}")

    history = state.get("chat_history", [])
    new_history = history + [
        {"role": "user", "content": state["query"]},
        {"role": "assistant", "content": result.answer},
    ]

    return {
        "answer": answer,
        "found": result.found,
        "citations": verified,
        "chat_history": new_history,
                "sources_used": [
            {
                "source": item.get("source"),
                "page": item.get("page"),
                "type": item.get("type", "text"),
                "image_path": item.get("image_path"),
            }
            for item in state["items"]
        ],
    }