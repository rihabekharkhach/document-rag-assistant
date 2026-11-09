import sqlite3
from contextlib import closing

import pytest

from app import db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.db")
    db.init_db()


def query(path, sql):
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(sql).fetchall()


def test_conversation_round_trip():
    cid = db.create_conversation("Ma discussion")
    db.add_message(cid, "user", "Bonjour ?")
    db.add_message(
        cid, "assistant", "Réponse", found=True,
        citations=[{"chunk": 1, "quote": "citation"}], standalone_query="Bonjour ?",
        sources=[{"source": "a.pdf", "page": 2, "type": "text", "image_path": None}],
        answer_mode="vision",
    )
    user, assistant = db.get_messages(cid)
    assert user["found"] is None and user["answer_mode"] == "text"
    assert assistant["found"] is True
    assert assistant["citations"] == [{"chunk": 1, "quote": "citation"}]
    assert assistant["sources"][0]["page"] == 2
    assert assistant["answer_mode"] == "vision"


def test_title_is_trimmed_and_defaulted():
    db.create_conversation("   ")
    long_id = db.create_conversation("x" * 200)
    titles = {c["id"]: c["title"] for c in db.list_conversations()}
    assert "Nouvelle conversation" in titles.values()
    assert len(titles[long_id]) == 80


def test_recent_history_returns_the_last_messages_oldest_first():
    cid = db.create_conversation("t")
    for i in range(10):
        db.add_message(cid, "user" if i % 2 == 0 else "assistant", f"m{i}")
    assert [m["content"] for m in db.recent_history(cid, limit=4)] == ["m6", "m7", "m8", "m9"]


def test_rename_and_delete_remove_the_messages_too():
    cid = db.create_conversation("old")
    assert db.rename_conversation(cid, "new")
    assert not db.rename_conversation(cid, "   ")
    assert db.list_conversations()[0]["title"] == "new"
    db.add_message(cid, "user", "hello")
    assert db.delete_conversation(cid)
    assert not db.delete_conversation(cid)
    assert not db.conversation_exists(cid)
    assert query(db.DB_PATH, "SELECT COUNT(*) FROM messages") == [(0,)]


def test_old_database_is_migrated(tmp_path, monkeypatch):
    old = tmp_path / "old.db"
    with closing(sqlite3.connect(old)) as conn:
        conn.executescript(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, "
            "created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, "
            "role TEXT NOT NULL, content TEXT NOT NULL, found INTEGER, citations TEXT, "
            "standalone_query TEXT, created_at TEXT NOT NULL);"
        )
    monkeypatch.setattr(db, "DB_PATH", old)
    db.init_db()
    columns = {row[1] for row in query(old, "PRAGMA table_info(messages)")}
    assert {"sources", "answer_mode"} <= columns