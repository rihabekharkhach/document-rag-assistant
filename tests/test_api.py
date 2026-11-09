from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db, main


class FakeGraph:
    def __init__(self):
        self.calls = []

    def invoke(self, state):
        self.calls.append(state)
        return {
            "answer": "Neuf conditions.", "found": True,
            "citations": [{"chunk": 1, "quote": "nine operating conditions"}],
            "standalone_query": state["query"],
            "sources_used": [{"source": "a.pdf", "page": 2, "type": "text", "image_path": None}],
            "answer_mode": "text",
        }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(main, "SOURCE_DIR", str(tmp_path / "docs"))
    graph = FakeGraph()
    monkeypatch.setattr(main, "research_graph", graph)
    with TestClient(main.app) as test_client:
        test_client.graph = graph
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}


def test_first_question_creates_a_conversation_and_stores_both_messages(client):
    body = client.post("/query", json={"question": "Combien de conditions ?", "title": "Mon test"}).json()
    assert body["answer"] == "Neuf conditions." and body["found"] is True
    assert body["sources"][0]["page"] == 2
    assert [c["title"] for c in client.get("/conversations").json()] == ["Mon test"]
    messages = client.get(f"/conversations/{body['thread_id']}").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_follow_up_receives_the_conversation_history(client):
    first = client.post("/query", json={"question": "Q1"}).json()
    client.post("/query", json={"question": "Et pour la validation ?", "thread_id": first["thread_id"]})
    history = client.graph.calls[-1]["chat_history"]
    assert [m["content"] for m in history] == ["Q1", "Neuf conditions."]


def test_unknown_thread_is_a_404(client):
    assert client.post("/query", json={"question": "Q", "thread_id": "nope"}).status_code == 404


def test_failed_pipeline_returns_502_and_leaves_no_empty_conversation(client, monkeypatch):
    class Broken:
        def invoke(self, state):
            raise RuntimeError("llm down")

    monkeypatch.setattr(main, "research_graph", Broken())
    assert client.post("/query", json={"question": "Q"}).status_code == 502
    assert client.get("/conversations").json() == []


def test_empty_question_is_rejected(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_rename_and_delete_conversation(client):
    thread = client.post("/query", json={"question": "Q"}).json()["thread_id"]
    assert client.patch(f"/conversations/{thread}", json={"title": "Renamed"}).status_code == 200
    assert client.get("/conversations").json()[0]["title"] == "Renamed"
    assert client.delete(f"/conversations/{thread}").status_code == 200
    assert client.get(f"/conversations/{thread}").status_code == 404


def test_upload_rejects_an_unsupported_type(client):
    response = client.post("/documents", files=[("files", ("notes.exe", b"x", "application/octet-stream"))])
    report = response.json()[0]
    assert report["status"] == "error" and "non supporté" in report["message"]


def test_upload_indexes_a_pdf_and_strips_folders_from_the_name(client, monkeypatch, tmp_path):
    seen = []

    def fake_ingest(path):
        seen.append(path)
        return {"name": Path(path).name, "status": "ok", "pages": 1, "characters": 10,
                "text_chunks": 1, "figures": 0, "seconds": 0.1, "message": ""}

    monkeypatch.setattr(main, "ingest_file", fake_ingest)
    report = client.post("/documents", files=[("files", ("../../evil.pdf", b"%PDF-1.4", "application/pdf"))]).json()[0]
    assert report["status"] == "ok" and report["name"] == "evil.pdf"
    assert Path(seen[0]).parent == tmp_path / "docs"


def test_documents_listing_and_deletion(client, monkeypatch):
    monkeypatch.setattr(main, "list_sources", lambda: [{"name": "a.pdf", "text_chunks": 3, "figures": 1}])
    deleted = []
    monkeypatch.setattr(main, "delete_source", lambda name: deleted.append(name))
    assert client.get("/documents").json() == [{"name": "a.pdf", "text_chunks": 3, "figures": 1}]
    assert client.delete("/documents/a.pdf").status_code == 200
    assert deleted == ["a.pdf"]


def test_image_endpoint_refuses_paths_outside_the_data_folders(client):
    for path in ("../../.env", "pyproject.toml", "C:/Windows/win.ini"):
        assert client.get("/image", params={"path": path}).status_code == 404


def test_page_image_for_an_unknown_pdf_is_a_404(client):
    assert client.get("/page-image", params={"source": "nothing.pdf", "page": 1}).status_code == 404