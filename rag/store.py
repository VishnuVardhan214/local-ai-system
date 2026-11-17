"""Local vector store: SQLite for text and metadata, numpy for the similarity search."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from rag.embeddings import tokenize

_STOP = set("the a an and or of to in on for with is are was were be what which who how why when where do does did "
            "this that these those it its as at by from about into than then so if not no can could should would".split())


class VectorStore:
    def __init__(self, db_path: str | Path):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self._cache: dict[str, tuple] = {}
        with self.lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS docs(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, path TEXT, sha TEXT UNIQUE,
                    added REAL, n_chunks INTEGER);
                CREATE TABLE IF NOT EXISTS chunks(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id INTEGER, idx INTEGER, locator TEXT,
                    text TEXT, embedder TEXT, vec BLOB);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """)
            self.conn.commit()

    def _q(self, sql, params=(), many=False):
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.fetchall() if many else cur

    def doc_by_sha(self, sha: str):
        rows = self._q("SELECT * FROM docs WHERE sha=?", (sha,), many=True)
        return dict(rows[0]) if rows else None

    def add_document(self, name: str, path: str, sha: str, chunks: list[dict], vectors: np.ndarray, embedder_id: str) -> int:
        with self.lock:
            cur = self.conn.execute("INSERT INTO docs(name,path,sha,added,n_chunks) VALUES(?,?,?,?,?)",
                                    (name, path, sha, time.time(), len(chunks)))
            doc_id = cur.lastrowid
            self.conn.executemany(
                "INSERT INTO chunks(doc_id,idx,locator,text,embedder,vec) VALUES(?,?,?,?,?,?)",
                [(doc_id, i, c["locator"], c["text"], embedder_id, vectors[i].astype(np.float32).tobytes())
                 for i, c in enumerate(chunks)])
            self.conn.commit()
        self._cache.clear()
        return doc_id

    def list_documents(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT id,name,n_chunks,added FROM docs ORDER BY id", many=True)]

    def delete_document(self, doc_id: int) -> bool:
        self._q("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        cur = self._q("DELETE FROM docs WHERE id=?", (doc_id,))
        self._cache.clear()
        return cur.rowcount > 0

    def chunk_count(self) -> int:
        return self._q("SELECT COUNT(*) AS n FROM chunks", many=True)[0]["n"]

    def embedders_in_use(self) -> list[str]:
        return [r["embedder"] for r in self._q(
            "SELECT embedder, COUNT(*) AS n FROM chunks GROUP BY embedder ORDER BY n DESC", many=True)]

    def all_chunks(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT id,text FROM chunks ORDER BY id", many=True)]

    def replace_vectors(self, ids: list[int], vectors: np.ndarray, embedder_id: str) -> None:
        with self.lock:
            self.conn.executemany("UPDATE chunks SET vec=?, embedder=? WHERE id=?",
                                  [(vectors[i].astype(np.float32).tobytes(), embedder_id, cid) for i, cid in enumerate(ids)])
            self.conn.commit()
        self._cache.clear()

    def _matrix(self, embedder_id: str):
        if embedder_id in self._cache:
            return self._cache[embedder_id]
        rows = self._q(
            "SELECT c.id, c.text, c.locator, c.vec, d.name FROM chunks c JOIN docs d ON d.id=c.doc_id "
            "WHERE c.embedder=? ORDER BY c.id", (embedder_id,), many=True)
        if not rows:
            self._cache[embedder_id] = (None, [])
            return self._cache[embedder_id]
        mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        meta = [{"id": r["id"], "text": r["text"], "locator": r["locator"], "doc": r["name"]} for r in rows]
        self._cache[embedder_id] = (mat, meta)
        return self._cache[embedder_id]

    def search(self, qvec: np.ndarray, qtext: str, embedder_id: str, k: int = 4, min_score: float = 0.0) -> list[dict]:
        mat, meta = self._matrix(embedder_id)
        if mat is None:
            return []
        sims = mat @ qvec.astype(np.float32)
        pool = np.argsort(-sims)[: max(k * 6, 20)]
        qwords = {w for w in tokenize(qtext) if w not in _STOP and len(w) > 1}
        scored = []
        for i in pool:
            words = set(tokenize(meta[i]["text"]))
            overlap = len(qwords & words) / len(qwords) if qwords else 0.0
            # Blend of meaning (vector) and exact words. Exact words help short queries a lot.
            score = 0.75 * float(sims[i]) + 0.25 * overlap
            scored.append((score, i))
        scored.sort(reverse=True)
        hits = []
        for score, i in scored[:k]:
            if score >= min_score:
                hits.append({**meta[i], "score": round(score, 3)})
        return hits
