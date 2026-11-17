from __future__ import annotations

from core.llm import OllamaClient
from core.mock import MockLLM


def build_llm(settings):
    kind = settings.backend.get("kind", "ollama")
    if kind == "mock":
        return MockLLM(settings=settings)
    return OllamaClient(settings)
