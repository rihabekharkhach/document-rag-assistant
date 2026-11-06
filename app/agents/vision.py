# app/agents/vision.py

from app.agents.state import GraphState
from app.rag.vision import ask_vision, pick_figures, should_use_vision


def route_after_generate(state: GraphState) -> str:
    return "vision" if should_use_vision(state) else "end"


def vision_node(state: GraphState) -> dict:
    print("[Vision] node triggered")
    figures = pick_figures(state)
    status, answer = ask_vision(state.get("standalone_query") or state["query"], figures)

    if status == "ok":
        return {
            "answer": answer,
            "found": True,
            "citations": [],  # no verbatim quote for a vision answer: the figure itself is the evidence
            "answer_mode": "vision",
            "sources_used": [
                {"source": f.get("source"), "page": f.get("page"), "type": "figure", "image_path": f.get("image_path")}
                for f in figures
            ],
        }

    if status == "not_found":
        # The model looked at the figure and could not answer. If the text answer rests ONLY on OCR text
        # read from figures (numbers whose position in the figure is lost), do not present it as a fact.
        items = state["items"]
        cited = [items[c["chunk"] - 1] for c in state.get("citations", []) if 0 < c["chunk"] <= len(items)]
        if cited and all(item.get("type") == "figure" for item in cited):
            return {
                "answer": "Je n'ai pas pu lire cette valeur de façon fiable sur la figure.",
                "found": False,
                "citations": [],
            }

    return {}  # vision failed, or the text answer has solid text support: keep it