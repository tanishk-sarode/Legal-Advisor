from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import sqlite3
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    scope_act TEXT NOT NULL DEFAULT 'All',
                    pinned INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_threads_user_updated
                ON threads(user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_thread_created
                ON messages(thread_id, created_at ASC);

                CREATE TABLE IF NOT EXISTS thread_memory (
                    thread_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
                );
                """
            )
            self._ensure_column(conn, "threads", "scope_act", "TEXT NOT NULL DEFAULT 'All'")
            self._ensure_column(conn, "threads", "pinned", "INTEGER NOT NULL DEFAULT 0")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        columns = {row[1] for row in rows}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def create_thread(
        self,
        user_id: str,
        thread_id: str,
        title: str = "New Chat",
        *,
        scope_act: str = "All",
    ) -> dict[str, Any]:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO threads(thread_id, user_id, title, scope_act, pinned, created_at, updated_at)
                VALUES (?, ?, ?, ?, 0, ?, ?)
                """,
                (thread_id, user_id, title, scope_act, now, now),
            )
            conn.execute(
                """
                INSERT INTO thread_memory(thread_id, summary, updated_at)
                VALUES (?, '', ?)
                ON CONFLICT(thread_id) DO NOTHING
                """,
                (thread_id, now),
            )
        return {
            "thread_id": thread_id,
            "user_id": user_id,
            "title": title,
            "scope_act": scope_act,
            "pinned": 0,
            "created_at": now,
            "updated_at": now,
        }

    def list_threads(self, user_id: str, search: str = "") -> list[dict[str, Any]]:
        query = """
            SELECT thread_id, user_id, title, scope_act, pinned, created_at, updated_at
            FROM threads
            WHERE user_id = ?
        """
        params: list[Any] = [user_id]
        if search.strip():
            query += " AND LOWER(title) LIKE ?"
            params.append(f"%{search.strip().lower()}%")
        query += " ORDER BY pinned DESC, updated_at DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_thread(self, user_id: str, thread_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT thread_id, user_id, title, scope_act, pinned, created_at, updated_at
                FROM threads
                WHERE user_id = ? AND thread_id = ?
                """,
                (user_id, thread_id),
            ).fetchone()
        return dict(row) if row else None

    def set_thread_scope(self, user_id: str, thread_id: str, scope_act: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE threads
                SET scope_act = ?, updated_at = ?
                WHERE user_id = ? AND thread_id = ?
                """,
                (scope_act, now, user_id, thread_id),
            )

    def set_thread_pinned(self, user_id: str, thread_id: str, pinned: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE threads
                SET pinned = ?
                WHERE user_id = ? AND thread_id = ?
                """,
                (1 if pinned else 0, user_id, thread_id),
            )

    def rename_thread(self, user_id: str, thread_id: str, title: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE threads
                SET title = ?, updated_at = ?
                WHERE user_id = ? AND thread_id = ?
                """,
                (title, now, user_id, thread_id),
            )

    def delete_thread(self, user_id: str, thread_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM threads WHERE user_id = ? AND thread_id = ?",
                (user_id, thread_id),
            )

    def touch_thread(self, thread_id: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                (now, thread_id),
            )

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        now = utc_now_iso()
        serialized_sources = json.dumps(sources or [], ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages(thread_id, role, content, sources_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (thread_id, role, content, serialized_sources, now),
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                (now, thread_id),
            )

    def get_messages(self, thread_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT message_id, thread_id, role, content, sources_json, created_at
                FROM messages
                WHERE thread_id = ?
                ORDER BY created_at ASC, message_id ASC
                """,
                (thread_id,),
            ).fetchall()

        messages: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            raw_sources = data.pop("sources_json")
            try:
                data["sources"] = json.loads(raw_sources) if raw_sources else []
            except json.JSONDecodeError:
                data["sources"] = []
            messages.append(data)
        return messages

    def get_summary(self, thread_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM thread_memory WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return row["summary"] if row else ""

    def set_summary(self, thread_id: str, summary: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO thread_memory(thread_id, summary, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id)
                DO UPDATE SET summary = excluded.summary, updated_at = excluded.updated_at
                """,
                (thread_id, summary, now),
            )

    def export_thread(self, user_id: str, thread_id: str) -> dict[str, Any] | None:
        thread = self.get_thread(user_id=user_id, thread_id=thread_id)
        if not thread:
            return None
        return {
            "thread": thread,
            "summary": self.get_summary(thread_id),
            "messages": self.get_messages(thread_id),
        }
