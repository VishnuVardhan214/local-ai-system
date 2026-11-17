"""Persistent memory in SQLite: conversations, notes, tasks and long-term facts."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path


class MemoryStore:
    def __init__(self, db_path: str | Path):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations(
                    id TEXT PRIMARY KEY, title TEXT, created REAL, updated REAL,
                    summary TEXT DEFAULT '', summarized_upto INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS messages(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, conv_id TEXT, role TEXT,
                    agent TEXT, content TEXT, meta TEXT, ts REAL);
                CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id, id);
                CREATE TABLE IF NOT EXISTS notes(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, text TEXT, tags TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS tasks(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, due TEXT, priority TEXT,
                    status TEXT DEFAULT 'open', created REAL, completed REAL);
                CREATE TABLE IF NOT EXISTS facts(
                    key TEXT PRIMARY KEY, value TEXT, updated REAL);
            """)
            self.conn.commit()

    def _run(self, sql: str, params=(), many=False):
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.fetchall() if many else cur

    # ---- conversations ----------------------------------------------------
    def create_conversation(self, title: str = "") -> str:
        cid = uuid.uuid4().hex[:12]
        now = time.time()
        self._run("INSERT INTO conversations(id,title,created,updated) VALUES(?,?,?,?)", (cid, title, now, now))
        return cid

    def conversation_exists(self, cid: str) -> bool:
        return bool(self._run("SELECT 1 FROM conversations WHERE id=?", (cid,), many=True))

    def list_conversations(self, limit: int = 50) -> list[dict]:
        rows = self._run("SELECT id,title,updated FROM conversations ORDER BY updated DESC LIMIT ?", (limit,), many=True)
        return [dict(r) for r in rows]

    def rename_conversation(self, cid: str, title: str) -> None:
        self._run("UPDATE conversations SET title=? WHERE id=?", (title[:80], cid))

    def delete_conversation(self, cid: str) -> None:
        self._run("DELETE FROM messages WHERE conv_id=?", (cid,))
        self._run("DELETE FROM conversations WHERE id=?", (cid,))

    def get_title(self, cid: str) -> str:
        rows = self._run("SELECT title FROM conversations WHERE id=?", (cid,), many=True)
        return rows[0]["title"] if rows else ""

    # ---- messages ---------------------------------------------------------
    def add_message(self, cid: str, role: str, content: str, agent: str = "", meta: dict | None = None) -> int:
        cur = self._run(
            "INSERT INTO messages(conv_id,role,agent,content,meta,ts) VALUES(?,?,?,?,?,?)",
            (cid, role, agent, content, json.dumps(meta or {}), time.time()),
        )
        self._run("UPDATE conversations SET updated=? WHERE id=?", (time.time(), cid))
        return cur.lastrowid

    def get_messages(self, cid: str) -> list[dict]:
        rows = self._run("SELECT * FROM messages WHERE conv_id=? ORDER BY id", (cid,), many=True)
        out = []
        for r in rows:
            d = dict(r)
            d["meta"] = json.loads(d["meta"] or "{}")
            out.append(d)
        return out

    def prompt_history(self, cid: str) -> tuple[str, list[dict]]:
        """Returns the running summary and the messages that are not yet summarized."""
        conv = self._run("SELECT summary,summarized_upto FROM conversations WHERE id=?", (cid,), many=True)
        if not conv:
            return "", []
        rows = self._run(
            "SELECT id,role,content FROM messages WHERE conv_id=? AND id>? ORDER BY id",
            (cid, conv[0]["summarized_upto"]), many=True)
        return conv[0]["summary"] or "", [dict(r) for r in rows]

    def messages_to_fold(self, cid: str, keep_last: int, threshold_chars: int) -> list[dict]:
        """Old messages that should be folded into the summary. Empty if not needed yet."""
        _, msgs = self.prompt_history(cid)
        if len(msgs) <= keep_last:
            return []
        old = msgs[:-keep_last]
        return old if sum(len(m["content"]) for m in msgs) > threshold_chars else []

    def set_summary(self, cid: str, summary: str, upto_id: int) -> None:
        self._run("UPDATE conversations SET summary=?, summarized_upto=? WHERE id=?", (summary, upto_id, cid))

    # ---- notes ------------------------------------------------------------
    def add_note(self, title: str, text: str, tags: str = "") -> int:
        return self._run("INSERT INTO notes(title,text,tags,created) VALUES(?,?,?,?)",
                         (title, text, tags, time.time())).lastrowid

    def search_notes(self, query: str = "", limit: int = 10) -> list[dict]:
        like = f"%{query}%"
        rows = self._run(
            "SELECT id,title,text,tags FROM notes WHERE title LIKE ? OR text LIKE ? OR tags LIKE ? "
            "ORDER BY id DESC LIMIT ?", (like, like, like, limit), many=True)
        return [dict(r) for r in rows]

    # ---- tasks ------------------------------------------------------------
    def add_task(self, title: str, due: str = "", priority: str = "normal") -> int:
        return self._run("INSERT INTO tasks(title,due,priority,created) VALUES(?,?,?,?)",
                         (title, due, priority, time.time())).lastrowid

    def list_tasks(self, status: str = "open") -> list[dict]:
        if status == "all":
            rows = self._run("SELECT * FROM tasks ORDER BY id", many=True)
        else:
            rows = self._run("SELECT * FROM tasks WHERE status=? ORDER BY id", (status,), many=True)
        return [dict(r) for r in rows]

    def complete_task(self, task_id: int) -> bool:
        cur = self._run("UPDATE tasks SET status='done', completed=? WHERE id=? AND status='open'",
                        (time.time(), task_id))
        return cur.rowcount > 0

    # ---- facts (long-term context) ----------------------------------------
    def remember(self, key: str, value: str) -> None:
        self._run("INSERT INTO facts(key,value,updated) VALUES(?,?,?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
                  (key.strip().lower(), value.strip(), time.time()))

    def recall(self, query: str = "", limit: int = 20) -> list[dict]:
        like = f"%{query.strip().lower()}%"
        rows = self._run("SELECT key,value FROM facts WHERE key LIKE ? OR lower(value) LIKE ? "
                         "ORDER BY updated DESC LIMIT ?", (like, like, limit), many=True)
        return [dict(r) for r in rows]

    def forget(self, key: str) -> bool:
        return self._run("DELETE FROM facts WHERE key=?", (key.strip().lower(),)).rowcount > 0
