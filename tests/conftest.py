import pytest

from config.settings import load_settings
from core.approvals import ApprovalManager
from core.mock import MockLLM
from memory.store import MemoryStore
from orchestrator import Orchestrator
from rag.service import RagService
from tools import ToolContext


def make_pdf(path, text):
    """Write a tiny one-page PDF so the PDF loader can be tested without extra libraries."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = "%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    path.write_bytes(out.encode("latin-1"))


@pytest.fixture
def settings(tmp_path):
    s = load_settings(overrides={
        "paths.data_dir": str(tmp_path / "data"),
        "paths.workspace_dirs": [str(tmp_path / "ws")],
        "backend.kind": "mock",
        "rag.embedder": "hashing",
        "security.approval_timeout_s": 5,
    })
    s.ensure_dirs()
    return s


@pytest.fixture
def memory(settings):
    return MemoryStore(settings.memory_db)


@pytest.fixture
def make_env(settings, memory):
    """Builds a full system around a scripted mock model."""
    def build(replies=None, approval="approve"):
        llm = MockLLM(replies=replies)
        rag = RagService(settings, llm)
        approvals = ApprovalManager(5, auto=approval)
        orch = Orchestrator(settings, llm, memory, rag, approvals)
        return orch, llm, rag, approvals
    return build


@pytest.fixture
def tool_ctx(settings, memory):
    llm = MockLLM()
    return ToolContext(settings=settings, memory=memory, rag=RagService(settings, llm),
                       workspace_dirs=settings.workspace_dirs)


def collect(gen):
    return list(gen)
