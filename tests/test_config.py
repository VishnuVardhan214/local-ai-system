import json
import re
import subprocess
import sys
from pathlib import Path

import httpx

from config.settings import ROOT, load_settings
from core.llm import OllamaClient


def test_env_variable_overrides_nested_values(monkeypatch, tmp_path):
    monkeypatch.setenv("LAI_GENERATION__NUM_CTX", "2048")
    monkeypatch.setenv("LAI_BACKEND__MODEL", "llama3.2:3b")
    monkeypatch.setenv("LAI_PATHS__WORKSPACE_DIRS", '["~/somewhere"]')
    monkeypatch.setenv("LAI_GENERATION__NUM_GPU", "")
    s = load_settings()
    assert s.generation.num_ctx == 2048 and s.backend.model == "llama3.2:3b"
    assert s.workspace_dirs[0].name == "somewhere"
    assert s.generation.num_gpu is None


def test_defaults_match_the_requested_conservative_setup():
    s = load_settings()
    assert s.generation.num_ctx == 4096
    assert s.ollama_server.OLLAMA_NUM_PARALLEL == "1"
    assert s.ollama_server.OLLAMA_FLASH_ATTENTION == "1"
    assert s.ollama_server.OLLAMA_KV_CACHE_TYPE == "q8_0"
    assert s.ui.host == "127.0.0.1"
    assert s.ollama_server.OLLAMA_HOST.startswith("127.0.0.1")


def test_use_mmap_option_is_only_sent_when_set(settings):
    seen = []

    def handler(req):
        seen.append(json.loads(req.content)["options"])
        done = {"done": True, "eval_count": 1, "eval_duration": 1, "message": {"content": "x"}}
        return httpx.Response(200, text=json.dumps(done) + "\n")

    c = OllamaClient(settings)
    c.http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    list(c.stream_chat([{"role": "user", "content": "hi"}]))
    settings._data["generation"]["use_mmap"] = False
    list(c.stream_chat([{"role": "user", "content": "hi"}]))
    assert "use_mmap" not in seen[0] and seen[1]["use_mmap"] is False


def test_ollama_env_helper_prints_exports():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "ollama_env.py"), "--print"],
                         capture_output=True, text=True, cwd=ROOT).stdout
    assert 'export OLLAMA_KV_CACHE_TYPE="q8_0"' in out and 'export OLLAMA_NUM_PARALLEL="1"' in out


def test_no_secrets_or_model_files_in_the_project():
    bad = [p for p in ROOT.rglob("*") if p.suffix in (".gguf", ".safetensors", ".bin") and "venv" not in p.parts]
    assert bad == []
    key_like = re.compile(r"sk-[A-Za-z0-9]{20,}|api_key\s*=\s*['\"][^'\"]+['\"]|AKIA[0-9A-Z]{16}")
    for p in ROOT.rglob("*.py"):
        assert not key_like.search(p.read_text(errors="ignore")), p
