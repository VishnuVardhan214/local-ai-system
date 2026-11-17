"""RAG service: ingest documents, search them, report status."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from core.llm import LLMError
from rag.chunking import chunk_segments
from rag.embeddings import HashingEmbedder, OllamaEmbedder
from rag.loaders import LoaderError, load_file
from rag.store import VectorStore


class RagService:
    def __init__(self, settings, llm):
        self.s = settings
        self.llm = llm
        self.store = VectorStore(settings.rag_db)
        self._embedder = None
        settings.docs_dir.mkdir(parents=True, exist_ok=True)

    # ---- embedder choice ----------------------------------------------------
    def _make(self, embedder_id: str):
        if embedder_id.startswith("hashing"):
            return HashingEmbedder()
        return OllamaEmbedder(self.llm)

    @property
    def embedder(self):
        if self._embedder is not None:
            return self._embedder
        mode = self.s.rag.get("embedder", "auto")
        if mode == "hashing":
            self._embedder = HashingEmbedder()
        elif mode == "ollama":
            self._embedder = OllamaEmbedder(self.llm)
        else:
            in_use = self.store.embedders_in_use()
            if in_use:  # never mix vectors from two embedders in one index
                self._embedder = self._make(in_use[0])
            else:
                try:
                    candidate = OllamaEmbedder(self.llm)
                    candidate.embed(["probe"])
                    self._embedder = candidate
                except (LLMError, Exception):
                    self._embedder = HashingEmbedder()
        return self._embedder

    def min_score(self) -> float:
        key = "min_score_hashing" if self.embedder.id.startswith("hashing") else "min_score_ollama"
        return float(self.s.rag.get(key, 0.12))

    # ---- ingest -------------------------------------------------------------
    def ingest_file(self, path: str | Path, display_name: str | None = None) -> dict:
        path = Path(path)
        name = display_name or path.name
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        existing = self.store.doc_by_sha(sha)
        if existing:
            return {"status": "duplicate", "name": existing["name"], "chunks": existing["n_chunks"], "id": existing["id"]}
        segments = load_file(path)
        chunks = chunk_segments(segments, int(self.s.rag.chunk_chars), int(self.s.rag.chunk_overlap))
        if not chunks:
            raise LoaderError("No text could be extracted from this file.")
        vectors = self.embedder.embed([c["text"] for c in chunks])
        doc_id = self.store.add_document(name, str(path), sha, chunks, vectors, self.embedder.id)
        return {"status": "added", "name": name, "chunks": len(chunks), "id": doc_id, "embedder": self.embedder.id}

    def ingest_upload(self, src: str | Path, original_name: str) -> dict:
        """Copy an uploaded file into the managed documents folder, then index it."""
        safe = Path(original_name).name
        dest = self.s.docs_dir / safe
        n = 1
        while dest.exists():
            dest = self.s.docs_dir / f"{Path(safe).stem}_{n}{Path(safe).suffix}"
            n += 1
        shutil.copyfile(src, dest)
        try:
            result = self.ingest_file(dest, display_name=safe)
        except Exception:
            dest.unlink(missing_ok=True)
            raise
        if result["status"] == "duplicate":
            dest.unlink(missing_ok=True)
        return result

    def reindex(self) -> int:
        """Re-embed everything with the current embedder (after changing the embedding setting)."""
        rows = self.store.all_chunks()
        if not rows:
            return 0
        vecs = self.embedder.embed([r["text"] for r in rows])
        self.store.replace_vectors([r["id"] for r in rows], vecs, self.embedder.id)
        return len(rows)

    # ---- query ----------------------------------------------------------------
    def search(self, query: str, k: int | None = None) -> list[dict]:
        if self.store.chunk_count() == 0:
            return []
        k = k or int(self.s.rag.top_k)
        qvec = self.embedder.embed([query])[0]
        return self.store.search(qvec, query, self.embedder.id, k=k, min_score=self.min_score())

    def documents(self) -> list[dict]:
        return self.store.list_documents()

    def delete(self, doc_id: int) -> bool:
        return self.store.delete_document(doc_id)

    def status(self) -> dict:
        return {"documents": len(self.documents()), "chunks": self.store.chunk_count(),
                "embedder": self.embedder.id if self.store.chunk_count() or self._embedder else self.s.rag.get("embedder", "auto")}
