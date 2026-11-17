"""Local web server. Binds to 127.0.0.1 by default and rejects foreign hosts and origins."""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from config.settings import ROOT, load_settings
from core import sysmon
from core.approvals import ApprovalManager
from core.factory import build_llm
from core.llm import LLMError
from memory.store import MemoryStore
from orchestrator import Orchestrator
from rag.loaders import SUPPORTED, LoaderError
from rag.service import RagService

FRONTEND = ROOT / "frontend"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str | None = None
    agent: str = "auto"


class ApproveRequest(BaseModel):
    id: str
    approve: bool


class ModelRequest(BaseModel):
    model: str


def _setup_logging(settings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(settings.log_dir / "app.log", maxBytes=1_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        root.addHandler(handler)
        root.setLevel(logging.INFO)


def create_app(settings=None, llm=None, approvals=None) -> FastAPI:
    settings = settings or load_settings()
    settings.ensure_dirs()
    _setup_logging(settings)
    llm = llm or build_llm(settings)
    approvals = approvals or ApprovalManager(float(settings.security.approval_timeout_s))
    memory = MemoryStore(settings.memory_db)
    rag = RagService(settings, llm)
    orch = Orchestrator(settings, llm, memory, rag, approvals)

    app = FastAPI(title="Local AI System", docs_url=None, redoc_url=None)
    allowed = list(settings.security.allowed_hosts)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed)

    @app.middleware("http")
    async def origin_guard(request: Request, call_next):
        """Block cross-site requests: a web page on another site must not drive this local app."""
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            host = urlparse(origin).hostname or ""
            if host not in ("localhost", "127.0.0.1", "::1", "testserver"):
                return JSONResponse({"detail": "Cross-site request blocked."}, status_code=403)
        return await call_next(request)

    app.state.orchestrator = orch
    app.state.llm = llm
    app.state.rag = rag
    app.state.memory = memory
    app.state.approvals = approvals

    # ---- pages ---------------------------------------------------------------
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True, "model_server_up": llm.is_up(), "model": llm.model, "num_ctx": llm.num_ctx}

    # ---- chat ----------------------------------------------------------------
    @app.post("/api/chat")
    def chat(req: ChatRequest):
        def stream():
            try:
                for ev in orch.handle(req.session_id, req.message.strip(), req.agent):
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
            except Exception as exc:
                logging.getLogger("server").exception("chat stream failed")
                yield f"data: {json.dumps({'type': 'error', 'message': type(exc).__name__})}\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/approve")
    def approve(req: ApproveRequest):
        if not approvals.resolve(req.id, req.approve):
            raise HTTPException(404, "No such pending approval.")
        return {"ok": True}

    @app.get("/api/state")
    def state():
        return {**orch.state, "pending_approvals": approvals.pending()}

    @app.get("/api/agents")
    def agents():
        return orch.agent_list()

    # ---- sessions --------------------------------------------------------------
    @app.get("/api/sessions")
    def sessions():
        return memory.list_conversations()

    @app.get("/api/sessions/{sid}")
    def session_messages(sid: str):
        if not memory.conversation_exists(sid):
            raise HTTPException(404, "Unknown session.")
        return memory.get_messages(sid)

    @app.delete("/api/sessions/{sid}")
    def delete_session(sid: str):
        memory.delete_conversation(sid)
        return {"ok": True}

    # ---- models ----------------------------------------------------------------
    @app.get("/api/models")
    def models():
        return {"current": llm.model, "num_ctx": llm.num_ctx, "available": llm.list_models(),
                "loaded": [m.get("name") for m in llm.ps()]}

    @app.post("/api/model")
    def set_model(req: ModelRequest):
        available = llm.list_models()
        if available and req.model not in available:
            raise HTTPException(404, "That model is not installed. Pull it with: ollama pull " + req.model)
        if not orch.gate.acquire(blocking=False):
            raise HTTPException(409, "A request is running. Try again when it finishes.")
        try:
            llm.set_model(req.model)
        finally:
            orch.gate.release()
        return {"current": llm.model}

    @app.post("/api/unload")
    def unload():
        llm.unload()
        return {"ok": True}

    # ---- documents ---------------------------------------------------------------
    @app.post("/api/upload")
    async def upload(files: list[UploadFile] = File(...)):
        limit = int(settings.ui.max_upload_mb) * 2**20
        results = []
        for f in files:
            name = Path(f.filename or "upload").name
            if Path(name).suffix.lower() not in SUPPORTED:
                results.append({"name": name, "status": "rejected", "error": "Unsupported file type."})
                continue
            tmp_dir = settings.data_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False, suffix=Path(name).suffix) as tmp:
                size = 0
                while chunk := await f.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        break
                    tmp.write(chunk)
            try:
                if size > limit:
                    results.append({"name": name, "status": "rejected", "error": "File is too large."})
                    continue
                results.append(rag.ingest_upload(tmp.name, name))
            except (LoaderError, LLMError) as exc:
                results.append({"name": name, "status": "failed", "error": str(exc)})
            except Exception:
                logging.getLogger("server").exception("ingest failed")
                results.append({"name": name, "status": "failed", "error": "Could not index this file."})
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        return results

    @app.get("/api/documents")
    def documents():
        return {"documents": rag.documents(), "status": rag.status()}

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: int):
        if not rag.delete(doc_id):
            raise HTTPException(404, "Unknown document.")
        return {"ok": True}

    # ---- monitor -------------------------------------------------------------------
    @app.get("/api/resources")
    def resources():
        snap = sysmon.snapshot()
        snap["loaded_models"] = [{"name": m.get("name"), "size_mb": round(m.get("size", 0) / 2**20),
                                  "vram_mb": round(m.get("size_vram", 0) / 2**20)} for m in llm.ps()]
        snap["state"] = orch.state
        return snap

    return app


def main() -> None:
    import uvicorn
    settings = load_settings()
    host, port = settings.ui.host, int(settings.ui.port)
    if host not in ("127.0.0.1", "localhost", "::1") and os.environ.get("LAI_ALLOW_REMOTE") != "1":
        sys.exit(f"Refusing to bind to {host}. This app is meant for localhost only. "
                 "Set LAI_ALLOW_REMOTE=1 if you really want this.")
    print(f"Local AI System on http://{host}:{port}  (model {settings.backend.model}, ctx {settings.generation.num_ctx})")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
