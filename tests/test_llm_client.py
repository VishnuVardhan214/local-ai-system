"""Tests the real Ollama client against a fake HTTP server, so the wire format is exercised."""
import json

import httpx
import pytest

from core.llm import LLMError, LLMUnavailable, OllamaClient


def ndjson(*objs):
    return "\n".join(json.dumps(o) for o in objs) + "\n"


def make_client(settings, handler):
    c = OllamaClient(settings)
    c.http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    return c


FINAL = {"done": True, "eval_count": 20, "eval_duration": 2_000_000_000, "prompt_eval_count": 100,
         "prompt_eval_duration": 500_000_000, "load_duration": 1_500_000_000}


def test_streaming_tokens_stats_and_request_shape(settings):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, text=ndjson(
            {"message": {"content": "Hel"}}, {"message": {"content": "lo"}}, {**FINAL, "message": {"content": ""}}))
    c = make_client(settings, handler)
    events = list(c.stream_chat([{"role": "user", "content": "hi"}]))
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Hello"
    stats = events[-1]["stats"]
    assert stats["tokens_per_s"] == 10.0 and stats["prompt_tokens_per_s"] == 200.0 and stats["load_s"] == 1.5
    body = seen["body"]
    assert body["options"]["num_ctx"] == 4096 and body["options"]["num_thread"] == 6
    assert body["think"] is False and body["keep_alive"] == "10m" and body["stream"] is True
    assert "num_gpu" not in body["options"]


def test_num_gpu_is_passed_when_configured(settings):
    seen = {}
    settings._data["generation"]["num_gpu"] = 20

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, text=ndjson({**FINAL, "message": {"content": "x"}}))
    list(make_client(settings, handler).stream_chat([{"role": "user", "content": "hi"}]))
    assert seen["body"]["options"]["num_gpu"] == 20


def test_tool_calls_are_parsed(settings):
    def handler(req):
        return httpx.Response(200, text=ndjson(
            {"message": {"content": "", "tool_calls": [{"function": {"name": "add_task", "arguments": {"title": "x"}}}]}},
            {**FINAL, "message": {"content": ""}}))
    out = make_client(settings, handler).chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert out["tool_calls"] == [{"name": "add_task", "arguments": {"title": "x"}}]


def test_thinking_tags_are_removed_from_plain_chat(settings):
    def handler(req):
        return httpx.Response(200, text=ndjson({"message": {"content": "<think>secret</think>Answer"}},
                                               {**FINAL, "message": {"content": ""}}))
    assert make_client(settings, handler).chat([{"role": "user", "content": "q"}])["content"] == "Answer"


def test_out_of_memory_steps_down_context_then_model(settings):
    attempts = []

    def handler(req):
        body = json.loads(req.content)
        attempts.append((body["model"], body["options"]["num_ctx"]))
        if body["model"] == "qwen3:4b":
            return httpx.Response(500, json={"error": "model requires more system memory than is available"})
        return httpx.Response(200, text=ndjson({"message": {"content": "ok"}}, {**FINAL, "message": {"content": ""}}))
    c = make_client(settings, handler)
    events = list(c.stream_chat([{"role": "user", "content": "hi"}]))
    assert attempts == [("qwen3:4b", 4096), ("qwen3:4b", 2048), ("llama3.2:3b", 2048)]
    assert sum(e["type"] == "fallback" for e in events) == 2
    assert (c.model, c.num_ctx) == ("llama3.2:3b", 2048)       # the working setup is kept
    attempts.clear()
    list(c.stream_chat([{"role": "user", "content": "again"}]))
    assert attempts == [("llama3.2:3b", 2048)]                  # and not retried from the top


def test_error_after_output_started_is_not_hidden(settings):
    def handler(req):
        return httpx.Response(200, text=ndjson({"message": {"content": "part"}}, {"error": "CUDA error: out of memory"}))
    with pytest.raises(LLMError):
        list(make_client(settings, handler).stream_chat([{"role": "user", "content": "hi"}]))


def test_fallback_can_be_disabled(settings):
    settings._data["fallback"]["enabled"] = False
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(500, json={"error": "out of memory"})
    with pytest.raises(LLMError):
        list(make_client(settings, handler).stream_chat([{"role": "user", "content": "hi"}]))
    assert len(calls) == 1


def test_server_down_gives_a_clear_error(settings):
    def handler(req):
        raise httpx.ConnectError("refused")
    with pytest.raises(LLMUnavailable, match="Is Ollama running"):
        list(make_client(settings, handler).stream_chat([{"role": "user", "content": "hi"}]))


def test_embed_runs_on_cpu(settings):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
    assert make_client(settings, handler).embed(["x"]) == [[0.1, 0.2]]
    assert seen["body"]["options"]["num_gpu"] == 0 and seen["body"]["model"] == "nomic-embed-text"


def test_housekeeping_calls(settings):
    def handler(req):
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "b"}, {"name": "a"}]})
        if req.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "a", "size": 1, "size_vram": 1}]})
        return httpx.Response(200, text="Ollama is running")
    c = make_client(settings, handler)
    assert c.list_models() == ["a", "b"] and c.ps()[0]["name"] == "a" and c.is_up()
