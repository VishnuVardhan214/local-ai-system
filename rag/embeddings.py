"""Embedders. The hashing one needs no model, so RAG still works if no embedding model is installed."""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class HashingEmbedder:
    """Feature hashing over words and word pairs. Fast, deterministic, no download."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.id = f"hashing-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = tokenize(text)
            grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
            for gram, count in Counter(grams).items():
                h = zlib.crc32(gram.encode("utf-8"))
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[row, h % self.dim] += sign * (1.0 + math.log(count))
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class OllamaEmbedder:
    def __init__(self, llm, batch: int = 16):
        self.llm = llm
        self.batch = batch
        self.id = f"ollama:{llm.embed_model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            vecs.extend(self.llm.embed(texts[i:i + self.batch]))
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
