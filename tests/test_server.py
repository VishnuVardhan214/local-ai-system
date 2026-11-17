import json

import pytest
from fastapi.testclient import TestClient

from app.server import create_app
from core.mock import MockLLM
from tests.conftest import make_pdf


@pytest.fixture
def client(settings):
    llm = MockLLM()
    return TestClient(create_app(settings, llm=llm)), llm


def sse(resp):
    return [json.loads(line[6:]) for line in resp.text.split("\n\n") if line.startswith("data: ")]


def test_health_and_page(client):
    c, _ = client
    assert c.get("/api/health").json()["ok"] is True
    page = c.get("/")
    assert page.status_code == 200 and "Local AI" in page.text
    assert c.get("/static/app.js").status_code == 200


def test_chat_stream_and_history(client):
    c, llm = client
    llm.replies = ["Hello from the mock."]
    r = c.post("/api/chat", json={"message": "hello there", "agent": "personal"})
    events = sse(r)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "session" and kinds[-1] == "done" and "route" in kinds
    sid = events[0]["id"]
    msgs = c.get(f"/api/sessions/{sid}").json()
    assert [m["role"] for m in msgs] == ["user", "assistant"] and "mock" in msgs[1]["content"]
    assert c.get("/api/sessions").json()[0]["id"] == sid
    assert c.delete(f"/api/sessions/{sid}").json()["ok"]
    assert c.get(f"/api/sessions/{sid}").status_code == 404


def test_empty_message_is_rejected(client):
    c, _ = client
    assert c.post("/api/chat", json={"message": ""}).status_code == 422


def test_upload_then_rag_answer(client, tmp_path):
    c, llm = client
    pdf = tmp_path / "report.pdf"; make_pdf(pdf, "Quarterly revenue grew thirty percent in Mumbai")
    txt = tmp_path / "bad.exe"; txt.write_bytes(b"MZ")
    r = c.post("/api/upload", files=[("files", ("report.pdf", pdf.read_bytes(), "application/pdf")),
                                     ("files", ("bad.exe", txt.read_bytes(), "application/octet-stream"))])
    out = r.json()
    assert out[0]["status"] == "added" and out[1]["status"] == "rejected"
    assert c.get("/api/documents").json()["documents"][0]["name"] == "report.pdf"
    llm.replies = ["Revenue grew thirty percent [1]."]
    events = sse(c.post("/api/chat", json={"message": "according to the document how did revenue change in Mumbai?"}))
    src = [e for e in events if e["type"] == "sources"][0]["items"][0]
    assert src["doc"] == "report.pdf" and src["locator"] == "page 1"


def test_upload_dedupes_and_can_delete(client, tmp_path):
    c, _ = client
    f = ("files", ("a.txt", b"same words in this file", "text/plain"))
    assert c.post("/api/upload", files=[f]).json()[0]["status"] == "added"
    assert c.post("/api/upload", files=[f]).json()[0]["status"] == "duplicate"
    doc = c.get("/api/documents").json()["documents"][0]
    assert c.delete(f"/api/documents/{doc['id']}").json()["ok"]
    assert c.get("/api/documents").json()["documents"] == []


def test_resources_report(client):
    c, _ = client
    r = c.get("/api/resources").json()
    assert r["ram_total_mb"] > 0 and "cpu_percent" in r and "state" in r and "gpu" in r


def test_model_listing_switching_and_unload(client):
    c, llm = client
    assert c.get("/api/models").json()["current"] == "mock-model"
    assert c.post("/api/model", json={"model": "mock-small"}).json()["current"] == "mock-small"
    assert c.post("/api/model", json={"model": "missing"}).status_code == 404
    assert c.post("/api/unload").json()["ok"] and llm.loaded is False


def test_agents_listed(client):
    c, _ = client
    names = {a["name"] for a in c.get("/api/agents").json()}
    assert {"coding", "research", "personal", "rag"} <= names


def test_foreign_host_header_is_rejected(client):
    c, _ = client
    assert c.get("/api/health", headers={"host": "evil.example.com"}).status_code == 400


def test_cross_site_post_is_rejected(client):
    c, _ = client
    r = c.post("/api/unload", headers={"origin": "https://evil.example.com"})
    assert r.status_code == 403
    assert c.post("/api/unload", headers={"origin": "http://localhost:8000"}).status_code == 200


def test_unknown_approval_id(client):
    c, _ = client
    assert c.post("/api/approve", json={"id": "nope", "approve": True}).status_code == 404
