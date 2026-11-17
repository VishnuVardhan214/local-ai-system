"""Read PDF, TXT, Markdown and DOCX files into (text, location) segments."""
from __future__ import annotations

from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".md", ".markdown", ".docx"}


class LoaderError(Exception):
    pass


def load_file(path: str | Path) -> list[tuple[str, str]]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise LoaderError(f"Unsupported file type: {ext or 'no extension'}")
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    if ext in (".md", ".markdown"):
        return _load_markdown(path)
    return [(path.read_text(encoding="utf-8", errors="replace"), "text")]


def _load_pdf(path: Path) -> list[tuple[str, str]]:
    from pypdf import PdfReader
    try:
        reader = PdfReader(str(path))
        segments = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                segments.append((text, f"page {i}"))
    except Exception as exc:  # pypdf raises many different error types
        raise LoaderError(f"Could not read the PDF: {exc}") from exc
    if not segments:
        raise LoaderError("The PDF has no extractable text. It may be a scan that needs OCR.")
    return segments


def _load_docx(path: Path) -> list[tuple[str, str]]:
    from docx import Document
    try:
        doc = Document(str(path))
    except Exception as exc:
        raise LoaderError(f"Could not read the DOCX: {exc}") from exc
    segments: list[tuple[str, str]] = []
    heading, buf = "start", []

    def flush():
        if buf:
            segments.append(("\n\n".join(buf), f"section: {heading}"))
            buf.clear()

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if (para.style.name or "").lower().startswith(("heading", "title")):
            flush()
            heading = text[:80]
        buf.append(text)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                buf.append(" | ".join(cells))
    flush()
    if not segments:
        raise LoaderError("The DOCX file has no text.")
    return segments


def _load_markdown(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    segments, heading, buf = [], "top", []
    for line in text.splitlines():
        if line.startswith("#"):
            if buf and "".join(buf).strip():
                segments.append(("\n".join(buf), f"section: {heading}"))
            heading, buf = line.lstrip("# ").strip()[:80] or "untitled", [line]
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        segments.append(("\n".join(buf), f"section: {heading}"))
    return segments
