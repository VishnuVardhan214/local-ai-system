"""One shared client for the local model server (Ollama).

Every agent talks to this single object, so only one model is ever loaded.
If the model fails to load (out of memory) before producing output, the client
steps down to a smaller context and then to the next fallback model.
"""
from __future__ import annotations

import json
import re
import time
from typing import Iterator

import httpx

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class LLMError(Exception):
    pass


class LLMUnavailable(LLMError):
    """The server cannot be reached."""


class LLMOutOfMemory(LLMError):
    """The model could not be loaded or run in the memory available."""


class LLMModelMissing(LLMError):
    """The model name is not installed."""


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def classify_error(status: int, body: str) -> LLMError:
    low = body.lower()
    memory_markers = (
        "out of memory", "more system memory", "unable to allocate", "cuda error",
        "runner process has terminated", "failed to allocate", "exit status",
    )
    if any(m in low for m in memory_markers):
        return LLMOutOfMemory(body[:300])
    if status == 404 or "not found" in low:
        return LLMModelMissing(body[:300])
    return LLMError(f"HTTP {status}: {body[:300]}")


def _stats_from(obj: dict, started: float, first_token_at: float | None) -> dict:
    eval_count = obj.get("eval_count", 0) or 0
    eval_ns = obj.get("eval_duration", 0) or 0
    prompt_ns = obj.get("prompt_eval_duration", 0) or 0
    prompt_count = obj.get("prompt_eval_count", 0) or 0
    return {
        "eval_count": eval_count,
        "tokens_per_s": round(eval_count / (eval_ns / 1e9), 2) if eval_ns else 0.0,
        "prompt_tokens": prompt_count,
        "prompt_tokens_per_s": round(prompt_count / (prompt_ns / 1e9), 2) if prompt_ns else 0.0,
        "load_s": round((obj.get("load_duration", 0) or 0) / 1e9, 3),
        "total_s": round(time.perf_counter() - started, 3),
        "ttft_s": round(first_token_at - started, 3) if first_token_at else None,
    }


class OllamaClient:
    def __init__(self, settings):
        self.s = settings
        self.base = settings.backend.host.rstrip("/")
        self.model = settings.backend.model
        self.embed_model = settings.backend.embed_model
        self.num_ctx = int(settings.generation.num_ctx)
        timeout = float(settings.backend.get("request_timeout_s", 600))
        self.http = httpx.Client(base_url=self.base, timeout=httpx.Timeout(timeout, connect=5.0))

    # ---- options -------------------------------------------------------
    def _options(self, ctx: int, num_predict: int | None, temperature: float | None) -> dict:
        g = self.s.generation
        opts = {
            "num_ctx": ctx,
            "temperature": g.temperature if temperature is None else temperature,
            "num_predict": num_predict or int(g.get("max_tokens", 1024)),
        }
        if g.get("num_thread"):
            opts["num_thread"] = int(g.num_thread)
        if g.get("num_gpu") is not None:
            opts["num_gpu"] = int(g.num_gpu)
        if g.get("use_mmap") is not None:
            opts["use_mmap"] = bool(g.use_mmap)
        return opts

    def _plan(self) -> list[tuple[str, int]]:
        """Ordered (model, context) attempts. The first entry is the current setup."""
        plan = [(self.model, self.num_ctx)]
        fb = self.s.fallback
        if not fb.get("enabled", True):
            return plan
        min_ctx = int(fb.get("min_ctx", 2048))
        if self.num_ctx // 2 >= min_ctx:
            plan.append((self.model, self.num_ctx // 2))
        for name in fb.get("models", []) or []:
            if name != self.model:
                plan.append((name, max(min_ctx, min(self.num_ctx, 2048))))
        return plan

    # ---- chat ----------------------------------------------------------
    def _stream_once(self, model, ctx, messages, tools, num_predict, temperature, think) -> Iterator[dict]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.s.generation.keep_alive,
            "options": self._options(ctx, num_predict, temperature),
        }
        if tools:
            payload["tools"] = tools
        if think is not None:
            payload["think"] = bool(think)

        for attempt in range(2):
            started = time.perf_counter()
            first = None
            try:
                with self.http.stream("POST", "/api/chat", json=payload) as r:
                    if r.status_code != 200:
                        body = r.read().decode("utf-8", "ignore")
                        # Older servers reject the think flag for models without thinking.
                        if attempt == 0 and "think" in body.lower() and "think" in payload:
                            payload.pop("think")
                            continue
                        raise classify_error(r.status_code, body)
                    for line in r.iter_lines():
                        if not line:
                            continue
                        obj = json.loads(line)
                        if "error" in obj:
                            raise classify_error(500, str(obj["error"]))
                        msg = obj.get("message") or {}
                        if msg.get("thinking"):
                            yield {"type": "thinking", "text": msg["thinking"]}
                        if msg.get("content"):
                            first = first or time.perf_counter()
                            yield {"type": "token", "text": msg["content"]}
                        for tc in msg.get("tool_calls") or []:
                            fn = tc.get("function", {})
                            args = fn.get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {}
                            yield {"type": "tool_call", "name": fn.get("name", ""), "arguments": args}
                        if obj.get("done"):
                            yield {"type": "done", "stats": _stats_from(obj, started, first)}
                return
            except httpx.ConnectError as exc:
                raise LLMUnavailable(f"Cannot reach the model server at {self.base}. Is Ollama running?") from exc
            except httpx.ReadTimeout as exc:
                raise LLMError("The model server took too long to answer.") from exc

    def stream_chat(self, messages, tools=None, num_predict=None, temperature=None, think=None) -> Iterator[dict]:
        if think is None:
            think = self.s.generation.get("think", False)
        last: LLMError | None = None
        for model, ctx in self._plan():
            started_output = False
            try:
                for ev in self._stream_once(model, ctx, messages, tools, num_predict, temperature, think):
                    started_output = True
                    yield ev
                if (model, ctx) != (self.model, self.num_ctx):
                    self.model, self.num_ctx = model, ctx  # keep the setup that worked
                return
            except (LLMOutOfMemory, LLMModelMissing) as exc:
                if started_output:
                    raise
                last = exc
                yield {"type": "fallback", "model": model, "ctx": ctx, "reason": str(exc)[:160]}
        raise last or LLMError("No usable model configuration.")

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

    def complete(self, prompt: str, system: str | None = None, num_predict: int = 256, temperature: float = 0.0) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, num_predict=num_predict, temperature=temperature, think=False)["content"]

    # ---- embeddings and housekeeping --------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeddings run on the CPU so they never compete with the chat model for VRAM."""
        try:
            r = self.http.post("/api/embed", json={
                "model": self.embed_model,
                "input": texts,
                "keep_alive": self.s.generation.keep_alive,
                "options": {"num_gpu": 0},
            })
        except httpx.ConnectError as exc:
            raise LLMUnavailable("Cannot reach the model server.") from exc
        if r.status_code != 200:
            raise classify_error(r.status_code, r.text)
        return r.json()["embeddings"]

    def is_up(self) -> bool:
        try:
            return self.http.get("/", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        try:
            r = self.http.get("/api/tags", timeout=5.0)
            return sorted(m["name"] for m in r.json().get("models", []))
        except (httpx.HTTPError, ValueError):
            return []

    def ps(self) -> list[dict]:
        try:
            return self.http.get("/api/ps", timeout=3.0).json().get("models", [])
        except (httpx.HTTPError, ValueError):
            return []

    def unload(self, model: str | None = None) -> None:
        try:
            self.http.post("/api/generate", json={"model": model or self.model, "keep_alive": 0}, timeout=10.0)
        except httpx.HTTPError:
            pass

    def set_model(self, name: str) -> None:
        old = self.model
        self.model = name
        if old != name:
            self.unload(old)
