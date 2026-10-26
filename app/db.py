# app/db.py

"""
Persistent storage for conversations and messages (SQLite, standard library only).
"""

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", "data/app.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    found INTEGER,
    citations TEXT,
    standalone_query TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect():
    """One short-lived connection per operation: safe with FastAPI's worker threads."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # needed for ON DELETE CASCADE
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


def create_conversation(title: str) -> str:
    conversation_id = str(uuid.uuid4())
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, title.strip()[:80] or "Nouvelle conversation", now, now),
        )
    return conversation_id


def conversation_exists(conversation_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    return row is not None


def list_conversations() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    found: bool | None = None,
    citations: list[dict] | None = None,
    standalone_query: str | None = None,
) -> None:
    now = _now()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO messages
               (conversation_id, role, content, found, citations, standalone_query, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                conversation_id, role, content,
                None if found is None else int(found),
                json.dumps(citations or [], ensure_ascii=False),
                standalone_query, now,
            ),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))


def get_messages(conversation_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT role, content, found, citations, standalone_query, created_at
               FROM messages WHERE conversation_id = ? ORDER BY id""",
            (conversation_id,),
        ).fetchall()
    return [
        {
            "role": r["role"],
            "content": r["content"],
            "found": None if r["found"] is None else bool(r["found"]),
            "citations": json.loads(r["citations"] or "[]"),
            "standalone_query": r["standalone_query"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def recent_history(conversation_id: str, limit: int = 6) -> list[dict]:
    """The last `limit` messages as {"role", "content"}, oldest first: what condense_question needs."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def delete_conversation(conversation_id: str) -> bool:
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    return cursor.rowcount > 0