# ui/app.py

"""
Streamlit interface.
  left bar : navigation (new discussion, knowledge base), theme switch, discussion history
  chat page: the discussion (bubbles, badges, sources, evidence viewer)
  knowledge page: documents, independent of any discussion

Run (the API must already be running):
    python -m streamlit run ui/app.py
"""

import os
from datetime import datetime, timezone
from urllib.parse import quote

import requests
import streamlit as st
from streamlit import config as st_config

try:
    import pymupdf  # only used to preview PDFs in the processing window
except ImportError:  # the preview then falls back to a short message
    pymupdf = None

APP_NAME = "Assistant documentaire"  # change the name here once you have chosen one
APP_ICON = "📄"

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
UPLOAD_TYPES = ["pdf", "docx", "png", "jpg", "jpeg"]
DEFAULT_PAGE_WIDTH = 420  # px, adjustable in the knowledge page settings
FIGURE_WIDTH = 600        # px

st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide")

# ---------------------------------------------------------------- themes

THEMES = {  # Streamlit's own theme options: every widget follows them
    "dark": {"theme.base": "dark", "theme.primaryColor": "#0891B2", "theme.backgroundColor": "#0B1020",
             "theme.secondaryBackgroundColor": "#111A2E", "theme.textColor": "#E6EAF2"},
    "light": {"theme.base": "light", "theme.primaryColor": "#0E7490", "theme.backgroundColor": "#F8FAFC",
              "theme.secondaryBackgroundColor": "#EEF2F7", "theme.textColor": "#0F172A"},
}

PALETTES = {  # colors of the parts drawn by this file's CSS
    "dark": {"accent": "#0891B2", "accent-soft": "#12324A", "accent-line": "#1F4E6E", "panel": "#111A2E",
             "line": "#24304A", "muted": "#94A3B8", "quote": "#CBD5E1",
             "hover": "rgba(255,255,255,.07)", "hover-strong": "rgba(255,255,255,.12)"},
    "light": {"accent": "#0E7490", "accent-soft": "#E0F2FE", "accent-line": "#BAE6FD", "panel": "#FFFFFF",
              "line": "#E2E8F0", "muted": "#64748B", "quote": "#334155",
              "hover": "rgba(15,23,42,.06)", "hover-strong": "rgba(15,23,42,.10)"},
}


def initial_mode() -> str:
    try:
        return "light" if st_config.get_option("theme.base") == "light" else "dark"
    except Exception:
        return "dark"


def apply_theme(mode: str) -> None:
    """Switch Streamlit's theme. Uses an internal setting that applies to the whole server:
    fine for a personal or demo deployment."""
    try:
        for option, value in THEMES[mode].items():
            st_config.set_option(option, value)
    except Exception:
        pass


def on_mode_change() -> None:
    mode = "light" if st.session_state.mode_toggle else "dark"
    st.session_state.mode = mode
    apply_theme(mode)


def palette_css(mode: str) -> str:
    variables = "".join(f"--{name}:{value};" for name, value in PALETTES[mode].items())
    return f"<style>:root{{{variables}}}</style>"


CSS = """
<style>
  /* the top band belongs to Streamlit's header: keep the content below it so buttons stay clickable */
  .block-container, .stMainBlockContainer { max-width: 1040px; padding-top: 4.5rem; padding-bottom: 7rem; }
  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }

  /* style-only elements take no room */
  .element-container:has(.dyn-style), [data-testid="stElementContainer"]:has(.dyn-style) { display: none; }

  /* left bar: plain rows, like ChatGPT / Claude */
  section[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
  section[data-testid="stSidebar"] .stButton > button,
  section[data-testid="stSidebar"] [data-testid^="stBaseButton"] {
    background: transparent; border: none; box-shadow: none; font-weight: 400;
    justify-content: flex-start !important; text-align: left !important;
    padding: .38rem .65rem; min-height: 2.1rem; border-radius: 9px; }
  section[data-testid="stSidebar"] [data-testid^="stBaseButton"] > div,
  section[data-testid="stSidebar"] [data-testid^="stBaseButton"] [data-testid="stMarkdownContainer"] {
    width: 100%; justify-content: flex-start !important; text-align: left !important; }
  section[data-testid="stSidebar"] [data-testid^="stBaseButton"] p {
    margin: 0; text-align: left !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  section[data-testid="stSidebar"] [data-testid^="stBaseButton"]:hover { background: var(--hover) !important; }
  section[data-testid="stSidebar"] [data-testid="stPopover"] button {
    background: transparent; border: none; box-shadow: none; color: var(--muted);
    padding: .2rem .45rem; min-height: 1.9rem; }
  section[data-testid="stSidebar"] [data-testid="stPopover"] button:hover { background: var(--hover-strong); }
  section[data-testid="stSidebar"] [data-testid="stPopover"] button [data-testid="stIconMaterial"] { display: none; }
  /* the ⋯ menu of a discussion appears on hover only */
  [class*="st-key-row_"] [data-testid="stPopover"] { opacity: 0; transition: opacity .12s; }
  [class*="st-key-row_"]:hover [data-testid="stPopover"] { opacity: 1; }

  /* drop zone */
  [data-testid="stFileUploaderDropzone"] { border: 1px dashed var(--accent-line); border-radius: 12px;
                                           background: transparent; padding: 1rem; }

  /* message bubbles (the marker span tells user and assistant messages apart) */
  .element-container:has(.msg-marker), [data-testid="stElementContainer"]:has(.msg-marker) { display: none; }
  [data-testid="stChatMessage"] { box-sizing: border-box; border: 1px solid var(--line); border-radius: 18px;
                                  padding: .9rem 1.1rem; margin-bottom: .7rem; background: var(--panel); }
  [data-testid="stChatMessage"]:has(.msg-user) { flex-direction: row-reverse; background: var(--accent-soft);
        border-color: var(--accent-line); width: fit-content; max-width: 78%; margin-left: auto; }
  [data-testid="stChatMessage"]:has(.msg-assistant) { width: 94%; margin-right: auto; }
  [data-testid="stChatMessageAvatarUser"] { background: var(--accent); }
  [data-testid="stChatMessageAvatarAssistant"] { background: #334155; }

  blockquote { border-left: 3px solid var(--accent); padding-left: .8rem; color: var(--quote); }
  [data-testid="stChatInput"] { border-radius: 16px; }
</style>
<span class="dyn-style"></span>
"""
st.markdown(CSS, unsafe_allow_html=True)

for key, value in {
    "view": "chat",                                              # "chat" or "knowledge"
    "mode": initial_mode(),                                      # "dark" or "light"
    "thread_id": None,
    "messages": [],
    "reports": [],
    "uploader_key": 0,
    "indexed_signature": None,
    "active": {},                                                # document name -> queried or not
    "settings": {"page_width": DEFAULT_PAGE_WIDTH, "show_rewritten": True},
}.items():
    st.session_state.setdefault(key, value)

st.markdown(palette_css(st.session_state.mode), unsafe_allow_html=True)


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
    health = requests.get(f"{API_URL}/health", timeout=5)
    health.raise_for_status()
except requests.RequestException:
    st.error("L'API est injoignable. Lancez-la avec : python -m uvicorn app.main:app --reload")
    st.stop()

if not health.json().get("qdrant", True):
    st.error("La base vectorielle (Qdrant) est arrêtée. Lancez Docker Desktop, puis : docker start qdrant")
    st.stop()


# ---------------------------------------------------------------- state helpers
# Widget state disappears when a widget is not rendered (for example when you change page),
# so what must persist lives in plain dicts, updated by callbacks.


def set_active(name: str) -> None:
    st.session_state.active[name] = st.session_state[f"use_{name}"]


def set_dialog_active(name: str) -> None:
    st.session_state.active[name] = st.session_state[f"dlg_{name}"]


def save_setting(name: str) -> None:
    st.session_state.settings[name] = st.session_state[f"setting_{name}"]


def active_documents(names: list[str]) -> list[str]:
    return [n for n in names if st.session_state.active.get(n, True)]


def page_width() -> int:
    return int(st.session_state.settings["page_width"])


def reset_conversation() -> None:
    st.session_state.thread_id = None
    st.session_state.messages = []


# ---------------------------------------------------------------- display helpers


def same_text(a: str, b: str) -> bool:
    keep = lambda t: "".join(ch for ch in t.lower() if ch.isalnum())
    return keep(a) == keep(b)


def source_at(sources: list[dict], chunk_number: int) -> dict | None:
    index = chunk_number - 1
    return sources[index] if 0 <= index < len(sources) else None


def source_label(src: dict) -> str:
    page = f", p.{src['page']}" if src.get("page") else ""
    return f"{src.get('source', '?')}{page}"


def file_icon(name: str) -> str:
    suffix = name.lower().rsplit(".", 1)[-1]
    return {"pdf": "📕", "docx": "📝"}.get(suffix, "🖼️")


def group_label(updated_at: str) -> str:
    try:
        when = datetime.fromisoformat(updated_at)
    except ValueError:
        return "Plus ancien"
    days = (datetime.now(timezone.utc).date() - when.astimezone(timezone.utc).date()).days
    if days <= 0:
        return "Aujourd'hui"
    if days == 1:
        return "Hier"
    if days <= 7:
        return "7 derniers jours"
    return "Plus ancien"


@st.cache_data(show_spinner=False)
def load_image(path: str) -> bytes | None:
    try:
        response = requests.get(f"{API_URL}/image", params={"path": path}, timeout=20)
        response.raise_for_status()
        return response.content
    except requests.RequestException:
        return None


@st.cache_data(show_spinner=False, max_entries=64)
def load_page(source: str, page: int, passage: str) -> bytes | None:
    try:
        response = requests.get(
            f"{API_URL}/page-image",
            params={"source": source, "page": page, "quote": passage},
            timeout=30,
        )
        response.raise_for_status()
        return response.content
    except requests.RequestException:
        return None


@st.cache_data(show_spinner=False, max_entries=16)
def pdf_page_count(data: bytes) -> int:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return len(doc)


@st.cache_data(show_spinner=False, max_entries=32)
def render_pdf_page(data: bytes, page: int) -> bytes:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return doc[page - 1].get_pixmap(dpi=100).tobytes("png")


def can_preview(src: dict) -> bool:
    return (
        src.get("type") != "figure"
        and bool(src.get("page"))
        and str(src.get("source", "")).lower().endswith(".pdf")
    )


def badge_line(message: dict, n_citations: int) -> str:
    if message.get("answer_mode") == "vision":
        return ":violet-badge[👁️ Lu sur la figure] :gray-badge[à vérifier sur l'image]"
    if message.get("found") is False:
        return ":orange-badge[⚠️ Non trouvé dans les documents]"
    if n_citations:
        return f":green-badge[✅ {n_citations} citation(s) vérifiée(s)]"
    return ":gray-badge[Sans citation vérifiée]"


# ---------------------------------------------------------------- windows (dialogs)


def preview_file(name: str, data: bytes) -> None:
    suffix = name.lower().rsplit(".", 1)[-1]
    if suffix in ("png", "jpg", "jpeg"):
        st.image(data, width=page_width())
    elif suffix == "pdf" and pymupdf is not None:
        total = pdf_page_count(data)
        page = st.slider("Page", 1, total, 1, key=f"slider_{name}_{len(data)}") if total > 1 else 1
        st.image(render_pdf_page(data, page), caption=f"Page {page} / {total}", width=page_width())
    else:
        st.info(f"{file_icon(name)} Aperçu non disponible pour ce type de fichier.")


@st.dialog("Traitement des documents", width="large")
def processing_dialog(files: list[tuple[str, bytes, str]]) -> None:
    """Left half: processing and report. Right half: the document itself."""
    signature = tuple((name, len(data)) for name, data, _ in files)
    left, right = st.columns(2, gap="large")

    with right:  # drawn first, so the document is visible while the processing runs
        st.markdown("##### Aperçu")
        names = [name for name, _, _ in files]
        chosen = st.selectbox("Document affiché", names, key="dialog_preview") if len(names) > 1 else names[0]
        name, data, _ = next(f for f in files if f[0] == chosen)
        preview_file(name, data)

    with left:
        st.markdown("##### Traitement")
        if st.session_state.indexed_signature != signature:
            with st.status("Analyse en cours…", expanded=True) as status:
                st.write("Extraction du texte, OCR des figures, découpage et indexation.")
                st.write("Cela peut prendre quelques minutes.")
                reports = api("POST", "/documents", timeout=900, files=[("files", f) for f in files])
                if reports is None:
                    status.update(label="Échec du traitement", state="error")
                else:
                    status.update(label="Traitement terminé", state="complete", expanded=False)
            if reports is not None:
                st.session_state.reports = reports
            st.session_state.indexed_signature = signature

        for report in st.session_state.reports:
            if report["status"] == "ok":
                st.success(
                    f"**{report['name']}**  \n{report['pages']} pages · {report['text_chunks']} extraits · "
                    f"{report['figures']} figures · {report['seconds']} s"
                )
                if report["message"]:
                    st.warning(report["message"])
            else:
                st.error(f"**{report['name']}** : {report['message']}")

        if st.button("Terminer", type="primary", use_container_width=True):
            st.session_state.uploader_key += 1
            st.session_state.reports = []
            st.session_state.indexed_signature = None
            st.rerun()  # closes the window and refreshes the lists


@st.dialog("Documents à interroger", width="small")
def choose_documents_dialog(names: list[str]) -> None:
    st.caption("Cochez les documents utilisés pour répondre.")
    c1, c2 = st.columns(2)
    if c1.button("Tout cocher", use_container_width=True):
        for n in names:
            st.session_state.active[n] = True
            st.session_state[f"dlg_{n}"] = True
    if c2.button("Tout décocher", use_container_width=True):
        for n in names:
            st.session_state.active[n] = False
            st.session_state[f"dlg_{n}"] = False

    for n in names:
        st.session_state.setdefault(f"dlg_{n}", st.session_state.active.get(n, True))
        st.checkbox(f"{file_icon(n)} {n}", key=f"dlg_{n}", on_change=set_dialog_active, args=(n,))

    st.divider()
    f1, f2 = st.columns(2)
    if f1.button("Terminé", type="primary", use_container_width=True):
        st.rerun()  # closes the window and refreshes the count on the chat page
    if f2.button("Ajouter…", use_container_width=True, help="Ouvrir la base de connaissances"):
        st.session_state.view = "knowledge"
        st.rerun()


# ---------------------------------------------------------------- left bar


def sidebar() -> None:
    view, current = st.session_state.view, st.session_state.thread_id

    with st.sidebar:
        st.markdown(f"#### {APP_ICON} {APP_NAME}")
        if st.button("＋  Nouvelle discussion", key="nav_new", use_container_width=True):
            reset_conversation()
            st.session_state.view = "chat"
            st.rerun()
        if st.button("📚  Base de connaissances", key="nav_knowledge", use_container_width=True):
            st.session_state.view = "knowledge"
            st.rerun()
        st.toggle("☀️ Mode clair", value=st.session_state.mode == "light", key="mode_toggle",
                  on_change=on_mode_change, help="Basculer entre le mode clair et le mode sombre")

        st.markdown("&nbsp;", unsafe_allow_html=True)
        search = st.text_input("Rechercher", placeholder="Rechercher une discussion",
                               label_visibility="collapsed", key="history_search").strip().lower()

        conversations = api("GET", "/conversations") or []
        conversations = [c for c in conversations if search in c["title"].lower()]
        if not conversations:
            st.caption("Aucune discussion." if not search else "Aucun résultat.")

        last_group = None
        for conv in conversations:
            group = group_label(conv["updated_at"])
            if group != last_group:
                st.caption(group.upper())
                last_group = group
            is_current = conv["id"] == current

            with st.container(key=f"row_{conv['id']}"):
                left, right = st.columns([8, 1.4], vertical_alignment="center", gap="small")
                if left.button(conv["title"], key=f"open_{conv['id']}", use_container_width=True):
                    messages = api("GET", f"/conversations/{conv['id']}")
                    if messages is not None:
                        st.session_state.thread_id = conv["id"]
                        st.session_state.messages = messages
                        st.session_state.view = "chat"
                        st.rerun()
                with right.popover("⋯"):
                    new_name = st.text_input("Nom", value=conv["title"], key=f"rn_{conv['id']}",
                                             label_visibility="collapsed", max_chars=80)
                    if st.button("✏️ Renommer", key=f"rnb_{conv['id']}", use_container_width=True):
                        if new_name.strip() and api("PATCH", f"/conversations/{conv['id']}",
                                                    json={"title": new_name}) is not None:
                            st.rerun()
                    st.divider()
                    st.caption("La suppression est définitive.")
                    if st.button("🗑 Supprimer", key=f"delc_{conv['id']}", use_container_width=True):
                        if api("DELETE", f"/conversations/{conv['id']}") is not None:
                            if is_current:
                                reset_conversation()
                            st.rerun()

    # highlight the open page / discussion
    highlighted = "nav_knowledge" if view == "knowledge" else (f"open_{current}" if current else "nav_new")
    st.markdown(
        f"<style>.st-key-{highlighted} button {{ background: rgba(8,145,178,.22) !important; }}</style>"
        '<span class="dyn-style"></span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- knowledge page


def knowledge_page(docs: list[dict]) -> None:
    st.markdown("## 📚 Base de connaissances")
    st.caption("Vos documents, indépendants des discussions : ajoutez-les ici, puis choisissez ceux à interroger.")

    if docs:
        c1, c2, c3 = st.columns(3)
        c1.metric("Documents", len(docs))
        c2.metric("Extraits de texte", sum(d["text_chunks"] for d in docs))
        c3.metric("Figures", sum(d["figures"] for d in docs))

    tab_docs, tab_settings = st.tabs([f"Documents ({len(docs)})", "Réglages"])

    with tab_docs:
        with st.container(border=True):
            st.markdown("**Ajouter des sources**")
            uploaded = st.file_uploader(
                "Déposer des fichiers", type=UPLOAD_TYPES, accept_multiple_files=True,
                key=f"uploader_{st.session_state.uploader_key}", label_visibility="collapsed",
            )
            if uploaded and st.button(f"Analyser {len(uploaded)} fichier(s)", type="primary"):
                st.session_state.reports = []
                st.session_state.indexed_signature = None
                processing_dialog([(f.name, f.getvalue(), f.type or "application/octet-stream") for f in uploaded])

        st.markdown(f"**Sources** `{len(docs)}`")
        if not docs:
            st.caption("Aucun document indexé.")
        for doc in docs:
            name = doc["name"]
            with st.container(border=True):
                c1, c2, c3 = st.columns([6, 1.7, 0.9], vertical_alignment="center")
                c1.markdown(f"{file_icon(name)} **{name}**")
                c1.caption(f"{doc['text_chunks']} extraits · {doc['figures']} figures · ✅ indexé")
                c2.toggle("Actif", value=st.session_state.active.get(name, True), key=f"use_{name}",
                          on_change=set_active, args=(name,), help="Interroger ce document dans les discussions")
                with c3.popover("⋯"):
                    st.caption("Supprimer ce document de la base ?")
                    if st.button("🗑 Supprimer", key=f"del_{name}", use_container_width=True):
                        if api("DELETE", f"/documents/{quote(name, safe='')}") is not None:
                            st.session_state.active.pop(name, None)
                            st.rerun()

    with tab_settings:
        settings = st.session_state.settings
        st.slider("Taille de l'aperçu des pages (px)", min_value=280, max_value=700, step=20,
                  value=settings["page_width"], key="setting_page_width",
                  on_change=save_setting, args=("page_width",))
        st.toggle("Afficher la question reformulée", value=settings["show_rewritten"],
                  key="setting_show_rewritten", on_change=save_setting, args=("show_rewritten",))
        st.caption("Astuce : survolez une image et cliquez sur l'icône ⛶ pour l'afficher en plein écran.")


# ---------------------------------------------------------------- chat page


def render_message(message: dict, previous_user: str, index: int) -> None:
    role = message["role"]
    with st.chat_message(role):
        st.markdown(f'<span class="msg-marker msg-{role}"></span>', unsafe_allow_html=True)
        if role != "assistant":
            st.markdown(message["content"])
            return

        sources = message.get("sources") or []
        citations = message.get("citations") or []

        st.markdown(badge_line(message, len(citations)))
        st.markdown(message["content"])

        if message.get("answer_mode") == "vision":
            figures: dict[str, dict] = {}
            for src in sources:
                if src.get("type") == "figure" and src.get("image_path"):
                    figures.setdefault(src["image_path"], src)
            for path, src in figures.items():
                data = load_image(path)
                if data:
                    st.image(data, caption=source_label(src), width=FIGURE_WIDTH)

        elif message.get("found") is not False:
            if citations:
                with st.expander(f"Sources ({len(citations)})"):
                    for j, c in enumerate(citations):
                        src = source_at(sources, c["chunk"])
                        with st.container(border=True):
                            st.markdown(f"> {c['quote']}")
                            if not src:
                                st.caption(f"extrait n°{c['chunk']}")
                                continue
                            st.caption(f"📄 {source_label(src)}")
                            if can_preview(src) and st.toggle(
                                "Voir la page avec le passage surligné", key=f"page_{index}_{j}"
                            ):
                                data = load_page(src["source"], int(src["page"]), c["quote"][:400])
                                if data:
                                    st.image(data, width=page_width())
                                else:
                                    st.caption("Page indisponible.")

            # A figure is shown only when it is the evidence: no text chunk supports the answer
            cited = [s for c in citations if (s := source_at(sources, c["chunk"])) is not None]
            if cited and not any(s.get("type") != "figure" for s in cited):
                figs: dict[str, dict] = {}
                for s in cited:
                    if s.get("image_path"):
                        figs.setdefault(s["image_path"], s)
                if figs:
                    with st.expander(f"Figure source ({len(figs)})", expanded=True):
                        for path, src in figs.items():
                            data = load_image(path)
                            if data:
                                st.image(data, caption=source_label(src), width=FIGURE_WIDTH)

        rewritten = message.get("standalone_query")
        if st.session_state.settings["show_rewritten"] and rewritten and not same_text(rewritten, previous_user):
            st.caption(f"Question reformulée : {rewritten}")


def chat_page(docs: list[dict]) -> None:
    names = [d["name"] for d in docs]
    selected = active_documents(names)

    if not names:
        with st.container(border=True):
            st.markdown("**Bienvenue.** Votre base de connaissances est vide : ajoutez d'abord des documents.")
            if st.button("📚 Ouvrir la base de connaissances", type="primary"):
                st.session_state.view = "knowledge"
                st.rerun()
        st.chat_input("Ajoutez d'abord des documents…", disabled=True)
        return

    left, right = st.columns([6, 1.4], vertical_alignment="center")
    left.caption(f"📚 {len(selected)}/{len(names)} documents actifs")
    if right.button("Gérer", key="manage_docs"):
        choose_documents_dialog(names)
    if not selected:
        st.warning("Aucun document actif : cliquez sur « Gérer » pour en activer au moins un.")

    if not st.session_state.messages:  # a new discussion: name it before the first question
        with st.container(border=True):
            st.markdown("### Nouvelle discussion")
            st.text_input("Nom de la discussion (facultatif)", key="new_title", max_chars=80,
                          placeholder="Ex. : Résultats de l'article sur les roulements")
            st.caption("Sans nom, la discussion porte le titre de votre première question. "
                       "Posez-la dans la barre en bas.")

    previous_user = ""
    for i, message in enumerate(st.session_state.messages):
        render_message(message, previous_user, i)
        if message["role"] == "user":
            previous_user = message["content"]

    question = st.chat_input("Posez votre question sur vos documents", disabled=not selected)
    if not (question and selected):
        return

    payload = {"question": question}
    if st.session_state.thread_id:
        payload["thread_id"] = st.session_state.thread_id
    else:
        title = st.session_state.get("new_title", "").strip()
        if title:
            payload["title"] = title
    if len(selected) < len(names):
        payload["sources"] = selected

    render_message({"role": "user", "content": question}, "", -1)  # shown while waiting
    with st.spinner("Recherche en cours…"):
        data = api("POST", "/query", timeout=180, json=payload)
    if data is not None:
        st.session_state.thread_id = data["thread_id"]
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({
            "role": "assistant", "content": data["answer"], "found": data["found"],
            "citations": data["citations"], "standalone_query": data["standalone_query"],
            "sources": data["sources"], "answer_mode": data.get("answer_mode", "text"),
        })
        st.rerun()


def main() -> None:
    docs = api("GET", "/documents") or []
    sidebar()
    if st.session_state.view == "knowledge":
        knowledge_page(docs)
    else:
        chat_page(docs)


main()