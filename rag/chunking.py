"""Split text segments into overlapping chunks that keep their location label."""
from __future__ import annotations

import re

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _tail(text: str, n: int) -> str:
    if n <= 0 or len(text) <= n:
        return text if n > 0 else ""
    cut = text[-n:]
    space = cut.find(" ")
    return cut[space + 1:] if space != -1 else cut


def _pieces(text: str, size: int) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"[ \t]+", " ", para).strip()
        if not para:
            continue
        if len(para) <= size:
            out.append(para)
            continue
        for sent in _SENTENCE.split(para):
            if len(sent) <= size:
                out.append(sent)
            else:
                out.extend(sent[i:i + size] for i in range(0, len(sent), size))
    return out


def chunk_segments(segments: list[tuple[str, str]], size: int = 900, overlap: int = 120) -> list[dict]:
    chunks: list[dict] = []
    for text, locator in segments:
        buf = ""
        for piece in _pieces(text, size):
            if buf and len(buf) + len(piece) + 1 > size:
                chunks.append({"text": buf.strip(), "locator": locator})
                buf = (_tail(buf, overlap) + " " + piece).strip()
            else:
                buf = f"{buf}\n{piece}" if buf else piece
        if buf.strip():
            chunks.append({"text": buf.strip(), "locator": locator})
    return chunks
