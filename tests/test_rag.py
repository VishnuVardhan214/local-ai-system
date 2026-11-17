import pytest
from docx import Document

from rag.chunking import chunk_segments
from rag.loaders import LoaderError, load_file
from tests.conftest import make_pdf


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i % 5}." for i in range(80))
    chunks = chunk_segments([(text, "page 1")], size=300, overlap=60)
    assert len(chunks) > 3
    assert all(len(c["text"]) <= 420 for c in chunks)
    assert all(c["locator"] == "page 1" for c in chunks)


def test_all_formats_load(tmp_path):
    (tmp_path / "a.txt").write_text("plain text about whales")
    (tmp_path / "b.md").write_text("# Intro\nhello\n## Details\nmore words here")
    d = Document(); d.add_heading("Budget", 1); d.add_paragraph("The budget is large."); d.save(tmp_path / "c.docx")
    make_pdf(tmp_path / "d.pdf", "Quarterly revenue grew strongly")
    assert load_file(tmp_path / "a.txt")[0][1] == "text"
    assert [loc for _, loc in load_file(tmp_path / "b.md")] == ["section: Intro", "section: Details"]
    assert load_file(tmp_path / "c.docx")[0][1] == "section: Budget"
    pdf = load_file(tmp_path / "d.pdf")
    assert pdf[0][1] == "page 1" and "revenue" in pdf[0][0]


def test_unsupported_and_empty(tmp_path):
    (tmp_path / "x.exe").write_bytes(b"MZ")
    with pytest.raises(LoaderError):
        load_file(tmp_path / "x.exe")
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    with pytest.raises(LoaderError):
        load_file(tmp_path / "bad.pdf")


def test_ingest_search_and_citation_metadata(tool_ctx, tmp_path):
    rag = tool_ctx.rag
    (tmp_path / "whales.txt").write_text("Blue whales are the largest animals. They eat krill in huge amounts.\n\n"
                                         "Their hearts weigh about as much as a small car.")
    (tmp_path / "cars.md").write_text("# Engines\nA four stroke engine has intake, compression, power and exhaust.")
    assert rag.ingest_file(tmp_path / "whales.txt")["status"] == "added"
    assert rag.ingest_file(tmp_path / "cars.md")["status"] == "added"
    hits = rag.search("what do blue whales eat")
    assert hits and hits[0]["doc"] == "whales.txt"
    hits = rag.search("how does a four stroke engine work")
    assert hits and hits[0]["doc"] == "cars.md" and hits[0]["locator"] == "section: Engines"


def test_duplicate_upload_is_detected(tool_ctx, tmp_path):
    f = tmp_path / "a.txt"; f.write_text("same content here")
    assert tool_ctx.rag.ingest_file(f)["status"] == "added"
    g = tmp_path / "b.txt"; g.write_text("same content here")
    assert tool_ctx.rag.ingest_file(g)["status"] == "duplicate"


def test_unrelated_query_finds_nothing(tool_ctx, tmp_path):
    f = tmp_path / "a.txt"; f.write_text("Blue whales eat krill.")
    tool_ctx.rag.ingest_file(f)
    assert tool_ctx.rag.search("quantum chromodynamics lagrangian") == []


def test_delete_and_reindex(tool_ctx, tmp_path):
    f = tmp_path / "a.txt"; f.write_text("Blue whales eat krill.")
    res = tool_ctx.rag.ingest_file(f)
    assert tool_ctx.rag.reindex() == 1
    assert tool_ctx.rag.delete(res["id"]) is True
    assert tool_ctx.rag.search("whales") == []
