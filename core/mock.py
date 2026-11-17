"""A stand-in model backend used by tests and for trying the UI without a model.

It can be given a script of replies. Each item is either a plain string (a text
answer) or a dict like {"tool_calls": [{"name": "read_file", "arguments": {...}}]}.
With no script it gives short deterministic answers.
"""
from __future__ import annotations

import re
from typing import Iterator

from core.llm import strip_think


class MockLLM:
    def __init__(self, replies: list | None = None, settings=None):
        self.model = "mock-model"
        self.num_ctx = 4096
        self.embed_model = "mock-embed"
        self.replies = list(replies or [])
        self.calls: list[dict] = []
        self.loaded = True

    def stream_chat(self, messages, tools=None, num_predict=None, temperature=None, think=None) -> Iterator[dict]:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        item = self.replies.pop(0) if self.replies else self._default(messages, tools)
        if isinstance(item, dict) and item.get("tool_calls"):
            for call in item["tool_calls"]:
                yield {"type": "tool_call", "name": call["name"], "arguments": call.get("arguments", {})}
            text = item.get("content", "")
        else:
            text = item if isinstance(item, str) else item.get("content", "")
        for word in re.findall(r"\S+\s*", text):
            yield {"type": "token", "text": word}
        n = len(text.split())
        yield {"type": "done", "stats": {
            "eval_count": n, "tokens_per_s": 100.0, "prompt_tokens": 50, "prompt_tokens_per_s": 500.0,
            "load_s": 0.0, "total_s": 0.01, "ttft_s": 0.001,
        }}

    def _default(self, messages, tools):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        last = messages[-1]["content"] if messages else ""
        if "Reply with only one label" in system:
            return "personal"
        if messages and messages[-1]["role"] == "tool":
            return "Done. Tool result was: " + last[:120]
        return "Mock answer to: " + last[-160:]

    def chat(self, messages, **kw) -> dict:
        text, calls, stats = [], [], {}
        for ev in self.stream_chat(messages, **kw):
            if ev["type"] == "token":
                text.append(ev["text"])
            elif ev["type"] == "tool_call":
                calls.append({"name": ev["name"], "arguments": ev["arguments"]})
            elif ev["type"] == "done":
                stats = ev["stats"]
        return {"content": strip_think("".join(text)), "tool_calls": calls, "stats": stats}

    def complete(self, prompt, system=None, num_predict=256, temperature=0.0) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.chat(msgs)["content"]

    def embed(self, texts):
        from rag.embeddings import HashingEmbedder
        return HashingEmbedder().embed(texts).tolist()

    def is_up(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["mock-model", "mock-small"]

    def ps(self) -> list[dict]:
        return [{"name": self.model, "size": 0, "size_vram": 0}] if self.loaded else []

    def unload(self, model=None) -> None:
        self.loaded = False

    def set_model(self, name: str) -> None:
        self.model = name
        self.loaded = True
