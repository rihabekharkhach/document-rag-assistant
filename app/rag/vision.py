# app/rag/vision.py

"""
Vision step: answer questions that depend on a figure by sending the real image to a
multimodal model. OCR stays the default; this runs only for visual questions, or when the
text answer was "not found" and a figure is the top result.
"""

import base64
import hashlib
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

VLM_MODEL = os.getenv("VLM_MODEL", "gemini-3.5-flash-lite")
VLM_RPM = float(os.getenv("VLM_RPM", "5"))
VISION_ENABLED = os.getenv("VISION_ENABLED", "true").lower() == "true"
MAX_IMAGES = 2   # images sent per question
CACHE_FILE = Path("data/vision_cache.json")
PROMPT_VERSION = "2"

VISUAL_WORDS = re.compile(
    r"\b(figures?|fig\.?|courbes?|barres?|matrices?|sch[eé]mas?|diagrammes?|graphiques?|histogrammes?|"
    r"images?|photos?|curves?|bars?|matrix|diagrams?|charts?|plots?|legend|l[eé]gende)\b",
    re.IGNORECASE,
)
FIGURE_REF = re.compile(r"\b(?:figures?|fig\.?)\s*(\d+)\b", re.IGNORECASE)

PROMPT = """You are looking at figure(s) from a document. Answer the question using ONLY what is visible in the image(s) and the printed text around them.
- Read values exactly as printed. Never guess or estimate a number you cannot read.
- The printed text around the figure (caption, legend, label order) tells you how to interpret class numbers or axis labels. For example, a legend listing names in class order maps class 0, 1, 2... to those names.
- If the information needed is not in the image(s) or in that text, reply with exactly: NOT_FOUND
- Answer in the same language as the question, in one or two sentences.

Figure captions: {captions}
Text printed around the figure(s): {nearby}
Question: {question}"""

_limiter = InMemoryRateLimiter(requests_per_second=VLM_RPM / 60, check_every_n_seconds=0.2, max_bucket_size=1)
_vlm = None


def _model() -> ChatGoogleGenerativeAI:
    """Created on first use, so the module imports fine even without a Google key."""
    global _vlm
    if _vlm is None:
        _vlm = ChatGoogleGenerativeAI(
            model=VLM_MODEL, temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=2, rate_limiter=_limiter,
        )
    return _vlm


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    

def _vision_file(item: dict) -> str:
    return item.get("vision_path") or item["image_path"]


def _usable(item: dict) -> bool:
    return item.get("type") == "figure" and bool(item.get("image_path")) and Path(_vision_file(item)).is_file()


def pick_figures(state: dict) -> list[dict]:
    """The figure(s) to show the vision model: the one the question names, else the top-ranked ones."""
    question = state.get("standalone_query") or state["query"]
    items = state.get("items") or []
    candidates = state.get("candidates") or items

    match = FIGURE_REF.search(question)
    if match:
        pattern = re.compile(rf"\bfigure\s+{match.group(1)}\b", re.IGNORECASE)
        for item in candidates:  # candidates are ordered best-first, so the first match is the best one
            if _usable(item) and pattern.search(item["text"][:260]):
                return [item]

    chosen, seen = [], set()
    for item in items:  # the whole final window, not only the top 3
        if _usable(item) and item["image_path"] not in seen:
            chosen.append(item)
            seen.add(item["image_path"])
        if len(chosen) == MAX_IMAGES:
            break
    return chosen


def should_use_vision(state: dict) -> bool:
    if not VISION_ENABLED:
        return False
    figures = pick_figures(state)
    question = state.get("standalone_query") or state["query"]
    visual = bool(VISUAL_WORDS.search(question))
    items = state.get("items") or []
    top_is_figure = bool(items) and items[0].get("type") == "figure"
    decision = bool(figures) and (visual or (not state.get("found", True) and top_is_figure))
    print(f"[Vision] check: visual={visual} figures_found={len(figures)} -> {'RUN' if decision else 'skip'}")
    return decision

def ask_vision(question: str, figures: list[dict]) -> tuple[str, str | None]:
    """Returns (status, answer): ("ok", text), ("not_found", None) or ("error", None)."""
    images = [Path(_vision_file(f)).read_bytes() for f in figures]
    nearby = " | ".join(f["nearby"] for f in figures if f.get("nearby"))[:600]
    key = hashlib.sha256(
        b"".join(images) + (question.strip().lower() + "|" + nearby + "|" + PROMPT_VERSION).encode("utf-8")
    ).hexdigest()
    cache = _load_cache()
    if key in cache:
        return ("ok", cache[key]) if cache[key] else ("not_found", None)

    captions = " | ".join(f["text"].splitlines()[0][:200] for f in figures)
    print(f"[Vision] asking about {len(figures)} figure(s): {captions[:100]!r}")
    print(f"[Vision] nearby text: {nearby[:150]!r}")
    content = [{"type": "text", "text": PROMPT.format(captions=captions, nearby=nearby or "(none)", question=question)}]
    for figure, data in zip(figures, images):
        mime = "image/png" if _vision_file(figure).lower().endswith(".png") else "image/jpeg"
        content.append({"type": "image_url", "image_url": f"data:{mime};base64,{base64.b64encode(data).decode()}"})

    try:
        reply = _model().invoke([HumanMessage(content=content)]).content
    except Exception as e:
        print(f"[Vision] failed ({type(e).__name__}): keeping the OCR-based answer")
        return ("error", None)

    if not isinstance(reply, str):  # some versions return a list of content blocks
        reply = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in reply)
    reply = reply.strip()
    print(f"[Vision] reply: {reply[:150]!r}")
    answer = None if (not reply or reply.upper().startswith("NOT_FOUND")) else reply
    cache[key] = answer or ""
    _save_cache(cache)
    return ("ok", answer) if answer else ("not_found", None)