# ui/app.py

"""
Streamlit interface, two screens:
  1. Documents: upload one or several files, see a short parsing report, manage the library.
  2. Conversation: chat with saved history and an optional document filter.

Run (the API must already be running):
    python -m streamlit run ui/app.py
"""

import os
from urllib.parse import quote

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
UPLOAD_TYPES = ["pdf", "docx", "png", "jpg", "jpeg"]

st.set_page_config(page_title="Assistant documentaire", page_icon="📄", layout="wide")

for key, value in {
    "page": "documents",   # first screen = documents
    "thread_id": None,
    "messages": [],
    "reports": [],
    "uploader_key": 0,     # changing it resets the file uploader after an upload
}.items():
    st.session_state.setdefault(key, value)

def same_text(a: str, b: str) -> bool:
    keep = lambda t: "".join(ch for ch in t.lower() if ch.isalnum())
    return keep(a) == keep(b)

def api(method: str, path: str, timeout: int = 120, **kwargs):
    """Call the backend. Returns the decoded JSON, or None after showing an error."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        st.error(f"Erreur lors de l'appel à l'API ({path}) : {error}")
        return None


try:
    requests.get(f"{API_URL}/health", timeout=3).raise_for_status()
except requests.RequestException:
    st.error("L'API est injoignable. Lancez-la avec : python -m uvicorn app.main:app --reload")
    st.stop()


# ---------------------------------------------------------------- screen 1


def documents_page() -> None:
    st.title("📄 Mes documents")
    st.caption("Joignez un ou plusieurs fichiers : chacun est analysé, découpé en extraits et indexé "
               "avant la conversation.")

    uploaded = st.file_uploader(
        "Joindre des fichiers", type=UPLOAD_TYPES, accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if st.button("Analyser et indexer", type="primary", disabled=not uploaded):
        files = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in uploaded]
        with st.spinner("Analyse en cours : extraction, découpage, OCR des figures, indexation..."):
            reports = api("POST", "/documents", timeout=900, files=files)
        if reports is not None:
            st.session_state.reports = reports
            st.session_state.uploader_key += 1  # clears the uploader
            st.rerun()

    for report in st.session_state.reports:
        with st.container(border=True):
            if report["status"] == "ok":
                st.success(f"**{report['name']}** : indexé")
                cols = st.columns(5)
                cols[0].metric("Pages / sections", report["pages"])
                cols[1].metric("Caractères", f"{report['characters']:,}".replace(",", " "))
                cols[2].metric("Extraits de texte", report["text_chunks"])
                cols[3].metric("Figures (OCR)", report["figures"])
                cols[4].metric("Durée (s)", report["seconds"])
                if report["message"]:
                    st.warning(report["message"])
            else:
                st.error(f"**{report['name']}** : {report['message']}")

    st.divider()
    st.subheader("Documents indexés")
    docs = api("GET", "/documents") or []
    if not docs:
        st.info("Aucun document indexé pour le moment.")
    for doc in docs:
        left, middle, right = st.columns([6, 3, 1])
        left.markdown(f"**{doc['name']}**")
        middle.caption(f"{doc['text_chunks']} extraits · {doc['figures']} figures")
        if right.button("🗑", key=f"del_{doc['name']}", help="Supprimer ce document"):
            if api("DELETE", f"/documents/{quote(doc['name'], safe='')}") is not None:
                st.rerun()

    st.divider()
    if st.button("Démarrer la conversation →", type="primary", disabled=not docs):
        st.session_state.page = "chat"
        st.rerun()


# ---------------------------------------------------------------- screen 2


def render_message(message: dict, previous_user: str) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] != "assistant":
            return
        if message.get("found") is False:
            st.caption("⚠️ Information non trouvée dans les documents.")
        elif message.get("citations"):
            with st.expander(f"Sources vérifiées ({len(message['citations'])})"):
                for c in message["citations"]:
                    st.markdown(f"> {c['quote']}")
                    st.caption(f"extrait n°{c['chunk']}")
        else:
            st.caption("ℹ️ Aucune citation vérifiée pour cette réponse.")
        rewritten = message.get("standalone_query")
        if rewritten and not same_text(rewritten, previous_user):
            st.caption(f"Question reformulée : {rewritten}")


def reset_conversation() -> None:
    st.session_state.thread_id = None
    st.session_state.messages = []


def chat_page() -> None:
    docs = api("GET", "/documents") or []
    names = [d["name"] for d in docs]

    with st.sidebar:
        if st.button("📎 Ajouter des documents", use_container_width=True):
            st.session_state.page = "documents"
            st.rerun()
        if st.button("➕ Nouvelle conversation", use_container_width=True):
            reset_conversation()
            st.rerun()

        selected = st.multiselect("Documents interrogés (vide = tous)", names)

        st.subheader("Historique")
        for conv in api("GET", "/conversations") or []:
            is_current = conv["id"] == st.session_state.thread_id
            title = conv["title"][:28] + ("…" if len(conv["title"]) > 28 else "")
            left, right = st.columns([5, 1])
            if left.button(("▶ " if is_current else "") + title, key=f"open_{conv['id']}",
                           use_container_width=True):
                messages = api("GET", f"/conversations/{conv['id']}")
                if messages is not None:
                    st.session_state.thread_id = conv["id"]
                    st.session_state.messages = messages
                    st.rerun()
            if right.button("🗑", key=f"delc_{conv['id']}", help="Supprimer cette conversation"):
                if api("DELETE", f"/conversations/{conv['id']}") is not None:
                    if is_current:
                        reset_conversation()
                    st.rerun()

    st.title("💬 Conversation")
    if not names:
        st.info("Aucun document indexé. Ajoutez des fichiers pour commencer.")
        st.stop()
    st.caption("Documents : " + (", ".join(selected) if selected else "tous les documents indexés"))

    previous_user = ""
    for message in st.session_state.messages:
        render_message(message, previous_user)
        if message["role"] == "user":
            previous_user = message["content"]

    if question := st.chat_input("Posez votre question sur vos documents"):
        payload = {"question": question}
        if st.session_state.thread_id:
            payload["thread_id"] = st.session_state.thread_id
        if selected:
            payload["sources"] = selected

        with st.spinner("Recherche en cours..."):
            data = api("POST", "/query", timeout=180, json=payload)
        if data is not None:
            st.session_state.thread_id = data["thread_id"]
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({
                "role": "assistant", "content": data["answer"], "found": data["found"],
                "citations": data["citations"], "standalone_query": data["standalone_query"],
            })
            st.rerun()


if st.session_state.page == "documents":
    documents_page()
else:
    chat_page()