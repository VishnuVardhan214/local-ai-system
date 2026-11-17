"""Benchmark and stability check for the local model setup.

Run on the laptop with Ollama running:

    python -m benchmarks.benchmark                 # measure the configured model
    python -m benchmarks.benchmark --auto-tune     # also step down if the setup is unstable
    python -m benchmarks.benchmark --model llama3.2:3b --ctx 4096

What it measures: cold load time, time to first token, generation speed, a long
context recall test, repeated requests (to spot memory growth), whether the
embedding model pushes the chat model out of memory, and real agent tool runs.
RAM, swap and VRAM are sampled in the background the whole time.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import load_settings  # noqa: E402
from core import sysmon  # noqa: E402
from core.approvals import ApprovalManager  # noqa: E402
from core.llm import LLMError, OllamaClient  # noqa: E402
from core.mock import MockLLM  # noqa: E402

RESULTS = ROOT / "benchmarks" / "results"

SIMPLE_PROMPT = "Explain in three sentences what a CPU cache is and why it makes programs faster."
REPEAT_PROMPTS = [
    "Give me two tips for writing readable Python functions.",
    "What is the difference between a process and a thread? Keep it short.",
    "List three common causes of a slow SQL query.",
    "Explain what a hash map is in two sentences.",
    "Why do we write unit tests? Answer in two sentences.",
    "What does the term 'idempotent' mean for an API? Be brief.",
    "Describe what a mutex does in one short paragraph.",
    "What is the purpose of an index in a database? Keep it brief.",
]


# ---------------------------------------------------------------------------
# Background sampler for RAM, swap and VRAM
# ---------------------------------------------------------------------------
class Sampler(threading.Thread):
    def __init__(self, interval: float = 0.4):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self._peak = {"model_rss_mb": 0.0, "ram_used_mb": 0, "swap_used_mb": 0, "vram_used_mb": 0,
                      "min_available_mb": 10 ** 9}

    def current(self) -> dict:
        vm, sw, gpu = psutil.virtual_memory(), psutil.swap_memory(), sysmon.read_gpu(max_age_s=0)
        return {"model_rss_mb": sysmon.model_process_rss_mb(),
                "ram_used_mb": round((vm.total - vm.available) / 2 ** 20),
                "available_mb": round(vm.available / 2 ** 20),
                "swap_used_mb": round(sw.used / 2 ** 20),
                "vram_used_mb": gpu["vram_used_mb"] if gpu else 0}

    def _update(self, cur: dict) -> None:
        with self._lock:
            p = self._peak
            p["model_rss_mb"] = max(p["model_rss_mb"], cur["model_rss_mb"])
            p["ram_used_mb"] = max(p["ram_used_mb"], cur["ram_used_mb"])
            p["swap_used_mb"] = max(p["swap_used_mb"], cur["swap_used_mb"])
            p["vram_used_mb"] = max(p["vram_used_mb"], cur["vram_used_mb"])
            p["min_available_mb"] = min(p["min_available_mb"], cur["available_mb"])

    def run(self) -> None:
        while not self._stop.is_set():
            self._update(self.current())
            self._stop.wait(self.interval)

    def take(self) -> dict:
        """Peaks since the last call, then start fresh. A final reading is added so
        a very short phase never returns an empty result."""
        self._update(self.current())
        with self._lock:
            out = dict(self._peak)
            self._reset()
        self._update(self.current())
        return out

    def finish(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Single requests
# ---------------------------------------------------------------------------
def ask(llm, prompt: str, num_predict: int = 128, system: str | None = None) -> dict:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    t0, first, text, stats = time.perf_counter(), None, [], {}
    for ev in llm.stream_chat(msgs, num_predict=num_predict, temperature=0.0, think=False):
        if ev["type"] == "token":
            first = first or time.perf_counter()
            text.append(ev["text"])
        elif ev["type"] == "done":
            stats = ev["stats"]
    wall = time.perf_counter() - t0
    gen_wall = wall - (first - t0) if first else 0
    return {
        "text": "".join(text),
        "ttft_s": round(first - t0, 3) if first else None,
        "wall_s": round(wall, 3),
        "tokens": stats.get("eval_count", 0),
        "tokens_per_s_server": stats.get("tokens_per_s", 0.0),
        "tokens_per_s_wall": round(stats.get("eval_count", 0) / gen_wall, 2) if gen_wall > 0.05 else 0.0,
        "prompt_tokens": stats.get("prompt_tokens", 0),
        "prompt_tokens_per_s": stats.get("prompt_tokens_per_s", 0.0),
        "load_s": stats.get("load_s", 0.0),
    }


def safe(fn, *a, **kw) -> dict:
    """Run a phase. Any failure is recorded instead of stopping the whole benchmark."""
    try:
        return fn(*a, **kw)
    except (LLMError, Exception) as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def phase_load(llm, sampler, mock: bool) -> dict:
    llm.unload()
    time.sleep(2.5)
    baseline = sampler.current()
    sampler.take()
    r = ask(llm, "Say hello.", num_predict=1)
    peaks = sampler.take()
    time.sleep(0.8)
    placement = []
    for m in llm.ps():
        size, vram = m.get("size", 0), m.get("size_vram", 0)
        placement.append({"name": m.get("name"), "size_mb": round(size / 2 ** 20), "on_gpu_mb": round(vram / 2 ** 20),
                          "gpu_fraction": round(vram / size, 2) if size else 0.0})
    return {"cold_load_wall_s": r["wall_s"], "load_s_server": r["load_s"], "baseline": baseline, "peaks": peaks,
            "placement": placement,
            "vram_added_mb": max(0, peaks["vram_used_mb"] - baseline["vram_used_mb"]),
            "ram_added_mb": max(0, peaks["ram_used_mb"] - baseline["ram_used_mb"])}


def phase_simple(llm, sampler) -> dict:
    ask(llm, "Warm up.", num_predict=4)   # make sure the model is warm
    sampler.take()
    r = ask(llm, SIMPLE_PROMPT, num_predict=160)
    r["peaks"] = sampler.take()
    r.pop("text", None)
    return r


def phase_long_context(llm, sampler, ctx: int, mock: bool) -> dict:
    """Fill about 70 percent of the context and ask for a fact hidden in the middle."""
    code = "MANGO-4417"
    para = ("Section {i}. The maintenance crew inspected valve {i} and recorded normal pressure, "
            "then moved on to the next station along the corridor. ")
    target_chars = int(ctx * 0.70 * 3.6)
    parts, i = [], 1
    while sum(len(p) for p in parts) < target_chars:
        parts.append(para.format(i=i))
        if sum(len(p) for p in parts) > target_chars / 2 and not any(code in p for p in parts):
            parts.append(f"IMPORTANT: the secret access code is {code}. ")
        i += 1
    doc = "".join(parts)
    sampler.take()
    r = ask(llm, f"{doc}\n\nQuestion: What is the secret access code? Answer with the code only.", num_predict=24)
    r["peaks"] = sampler.take()
    r["recalled_code"] = code in r.pop("text", "") if not mock else None
    r["target_prompt_tokens_approx"] = int(ctx * 0.70)
    return r


def phase_repeated(llm, sampler, n: int) -> dict:
    rows = []
    sampler.take()
    for i in range(n):
        r = ask(llm, REPEAT_PROMPTS[i % len(REPEAT_PROMPTS)], num_predict=96)
        time.sleep(0.5)
        cur = sampler.current()
        rows.append({"i": i + 1, "ttft_s": r["ttft_s"], "tokens_per_s": r["tokens_per_s_server"],
                     "model_rss_mb": cur["model_rss_mb"], "ram_used_mb": cur["ram_used_mb"],
                     "swap_used_mb": cur["swap_used_mb"], "vram_used_mb": cur["vram_used_mb"]})
    peaks = sampler.take()
    half = max(1, n // 2)
    return {"runs": rows, "peaks": peaks,
            "rss_growth_mb": round(rows[-1]["model_rss_mb"] - rows[0]["model_rss_mb"], 1),
            "vram_growth_mb": rows[-1]["vram_used_mb"] - rows[0]["vram_used_mb"],
            "ram_growth_mb": rows[-1]["ram_used_mb"] - rows[0]["ram_used_mb"],
            "avg_tokens_per_s": round(sum(x["tokens_per_s"] for x in rows) / n, 2),
            "avg_ttft_s": round(sum(x["ttft_s"] or 0 for x in rows) / n, 3),
            "speed_first_half": round(sum(x["tokens_per_s"] for x in rows[:half]) / half, 2),
            "speed_second_half": round(sum(x["tokens_per_s"] for x in rows[half:]) / max(1, n - half), 2)}


def phase_embed_eviction(llm) -> dict:
    """If an embedding call evicts the chat model, every RAG question pays a full reload."""
    ask(llm, "Warm up.", num_predict=2)
    t0 = time.perf_counter()
    llm.embed(["test sentence for the embedding model"])
    embed_s = time.perf_counter() - t0
    r = ask(llm, "Say hi.", num_predict=2)
    loaded = [m.get("name") for m in llm.ps()]
    return {"embed_call_s": round(embed_s, 2), "chat_reload_s_after_embed": r["load_s"],
            "models_loaded_together": loaded, "chat_model_was_evicted": r["load_s"] > 1.0}


def phase_agents(llm) -> dict:
    """Run real agent tasks through the orchestrator and check the outcome."""
    from memory.store import MemoryStore
    from orchestrator import Orchestrator
    from rag.service import RagService

    tmp = Path(tempfile.mkdtemp(prefix="lai_bench_"))
    try:
        s = load_settings(overrides={"paths.data_dir": str(tmp / "data"), "paths.workspace_dirs": [str(tmp / "ws")],
                                     "rag.embedder": "hashing"})
        s.ensure_dirs()
        memory, rag = MemoryStore(s.memory_db), RagService(s, llm)
        orch = Orchestrator(s, llm, memory, rag, ApprovalManager(10, auto="approve"))
        (s.workspace_dirs[0] / "data.txt").write_text("The magic number is 8841.\n")
        (tmp / "whales.txt").write_text("Blue whales are the largest animals. They eat tiny shrimp called krill.")
        rag.ingest_file(tmp / "whales.txt")

        cases = [
            ("personal task tool", "personal", "Add a task: buy milk tomorrow",
             lambda ans, m: any("milk" in t["title"].lower() for t in m.list_tasks())),
            ("coding file tool", "coding", "Read the file data.txt and tell me the magic number.",
             lambda ans, m: "8841" in ans),
            ("rag with sources", "rag", "What do blue whales eat?",
             lambda ans, m: "krill" in ans.lower()),
        ]
        results = []
        for name, agent, prompt, check in cases:
            t0, answer, tools, err, sources = time.perf_counter(), [], 0, None, 0
            for ev in orch.handle(None, prompt, agent):
                if ev["type"] == "token":
                    answer.append(ev["text"])
                elif ev["type"] == "tool_call":
                    tools += 1
                elif ev["type"] == "sources":
                    sources = len(ev["items"])
                elif ev["type"] == "error":
                    err = ev["message"]
            text = "".join(answer)
            results.append({"case": name, "ok": bool(check(text, memory)) and not err, "seconds": round(time.perf_counter() - t0, 1),
                            "tool_calls": tools, "sources": sources, "error": err})
        return {"cases": results, "passed": sum(r["ok"] for r in results), "total": len(results)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# One full run and the stability verdict
# ---------------------------------------------------------------------------
def verdict(rep: dict, args) -> dict:
    problems, warnings = [], []
    for name, ph in rep["phases"].items():
        if isinstance(ph, dict) and ph.get("error"):
            problems.append(f"{name} failed: {ph['error']}")
    swap_base = rep["phases"].get("load", {}).get("baseline", {}).get("swap_used_mb", 0)
    swap_peak = max((ph.get("peaks", {}).get("swap_used_mb", 0) for ph in rep["phases"].values() if isinstance(ph, dict)), default=0)
    if swap_peak - swap_base > args.max_swap_growth_mb:
        problems.append(f"swap grew by {swap_peak - swap_base} MB")
    min_avail = min((ph.get("peaks", {}).get("min_available_mb", 10 ** 9) for ph in rep["phases"].values() if isinstance(ph, dict)), default=10 ** 9)
    if min_avail < args.min_free_ram_mb:
        problems.append(f"free RAM dropped to {min_avail} MB")
    rep_ph = rep["phases"].get("repeated", {})
    if rep_ph.get("rss_growth_mb", 0) > args.max_growth_mb:
        problems.append(f"model process memory grew {rep_ph['rss_growth_mb']} MB over repeated requests")
    if rep_ph.get("vram_growth_mb", 0) > 200:
        problems.append(f"VRAM grew {rep_ph['vram_growth_mb']} MB over repeated requests")
    gpu = rep["meta"].get("gpu")
    vram_peak = max((ph.get("peaks", {}).get("vram_used_mb", 0) for ph in rep["phases"].values() if isinstance(ph, dict)), default=0)
    if gpu and vram_peak > gpu["vram_total_mb"] - 150:
        problems.append(f"VRAM peaked at {vram_peak} of {gpu['vram_total_mb']} MB (no safety margin)")
    tps = rep_ph.get("avg_tokens_per_s", 0)
    if tps and tps < args.min_tps:
        problems.append(f"generation is slow ({tps} tok/s, wanted at least {args.min_tps})")
    long_ph = rep["phases"].get("long_context", {})
    if long_ph.get("recalled_code") is False:
        warnings.append("the model did not find the hidden code in the long prompt")
    if rep["phases"].get("embed", {}).get("chat_model_was_evicted"):
        warnings.append("embedding evicted the chat model; set RAG embedder to hashing or raise OLLAMA_MAX_LOADED_MODELS")
    ag = rep["phases"].get("agents", {})
    if ag.get("total") and ag["passed"] < ag["total"]:
        warnings.append(f"agent checks passed {ag['passed']} of {ag['total']}")
    placement = rep["phases"].get("load", {}).get("placement", [])
    for p in placement:
        if 0 < p["gpu_fraction"] < 0.99 and "embed" not in (p["name"] or ""):
            warnings.append(f"{p['name']} is only {int(p['gpu_fraction'] * 100)} percent on the GPU, the rest runs on the CPU")
    return {"stable": not problems, "problems": problems, "warnings": warnings}


def run_once(settings, model: str, ctx: int, args, mock: bool) -> dict:
    sampler = Sampler()
    sampler.start()
    gpu = sysmon.read_gpu(max_age_s=0)
    meta = {"model": model, "num_ctx": ctx, "num_thread": settings.generation.get("num_thread"),
            "num_gpu": settings.generation.get("num_gpu"), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": platform.python_version(), "os": platform.platform(), "gpu": gpu, "mock": mock,
            "ram_total_mb": round(psutil.virtual_memory().total / 2 ** 20)}
    if mock:
        llm = MockLLM()
        llm.model, llm.num_ctx = model, ctx
    else:
        settings._data["backend"]["model"] = model
        settings._data["generation"]["num_ctx"] = ctx
        settings._data["fallback"]["enabled"] = False   # this script does its own stepping down
        llm = OllamaClient(settings)
    rep = {"meta": meta, "phases": {}}
    ph = rep["phases"]
    try:
        ph["load"] = safe(phase_load, llm, sampler, mock)
        if not ph["load"].get("error"):
            ph["simple"] = safe(phase_simple, llm, sampler)
            if not args.skip_long:
                ph["long_context"] = safe(phase_long_context, llm, sampler, ctx, mock)
            ph["repeated"] = safe(phase_repeated, llm, sampler, args.repeats)
            if not mock:
                ph["embed"] = safe(phase_embed_eviction, llm)
                if not args.skip_agents:
                    ph["agents"] = safe(phase_agents, llm)
    finally:
        sampler.finish()
    rep["verdict"] = verdict(rep, args)
    return rep


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_report(rep: dict) -> None:
    m, ph, v = rep["meta"], rep["phases"], rep["verdict"]
    line = "=" * 64
    print(f"\n{line}\n {m['model']}  ctx {m['num_ctx']}  threads {m['num_thread']}  gpu layers {m['num_gpu'] or 'auto'}"
          f"{'  (MOCK, numbers are not real)' if m['mock'] else ''}\n{line}")
    ld = ph.get("load", {})
    if ld.get("error"):
        print(" LOAD FAILED:", ld["error"])
    else:
        print(f" cold load            {ld['cold_load_wall_s']} s   (server reports {ld['load_s_server']} s)")
        for p in ld["placement"]:
            print(f" placement            {p['name']}: {p['size_mb']} MB total, {p['on_gpu_mb']} MB on GPU ({int(p['gpu_fraction'] * 100)} percent)")
        print(f" added by the model   RAM +{ld['ram_added_mb']} MB, VRAM +{ld['vram_added_mb']} MB")
    s = ph.get("simple", {})
    if s and not s.get("error"):
        print(f" first token          {s['ttft_s']} s")
        print(f" generation speed     {s['tokens_per_s_server']} tok/s (wall clock {s['tokens_per_s_wall']})")
        print(f" peak model RSS       {s['peaks']['model_rss_mb']} MB   peak VRAM {s['peaks']['vram_used_mb']} MB")
    lc = ph.get("long_context", {})
    if lc and not lc.get("error"):
        print(f" long context         {lc['prompt_tokens']} prompt tokens, first token {lc['ttft_s']} s, "
              f"prompt speed {lc['prompt_tokens_per_s']} tok/s, found hidden code: {lc['recalled_code']}")
        print(f"                      peak model RSS {lc['peaks']['model_rss_mb']} MB, peak VRAM {lc['peaks']['vram_used_mb']} MB, lowest free RAM {lc['peaks']['min_available_mb']} MB")
    rp = ph.get("repeated", {})
    if rp and not rp.get("error"):
        print(f" repeated x{len(rp['runs'])}        avg {rp['avg_tokens_per_s']} tok/s, avg first token {rp['avg_ttft_s']} s, "
              f"speed {rp['speed_first_half']} -> {rp['speed_second_half']}")
        print(f"                      model memory drift {rp['rss_growth_mb']} MB, VRAM drift {rp['vram_growth_mb']} MB, "
              f"peak swap {rp['peaks']['swap_used_mb']} MB")
    em = ph.get("embed", {})
    if em and not em.get("error"):
        print(f" embed then chat      chat reload {em['chat_reload_s_after_embed']} s, evicted: {em['chat_model_was_evicted']}, "
              f"loaded together: {em['models_loaded_together']}")
    ag = ph.get("agents", {})
    if ag and not ag.get("error"):
        for c in ag["cases"]:
            print(f" agent {c['case']:<20} {'PASS' if c['ok'] else 'FAIL'}  {c['seconds']} s, {c['tool_calls']} tool calls")
    print(line)
    print(" RESULT:", "STABLE" if v["stable"] else "UNSTABLE")
    for p in v["problems"]:
        print("   problem:", p)
    for w in v["warnings"]:
        print("   note:   ", w)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--auto-tune", action="store_true", help="step down to a smaller context or model if unstable")
    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--skip-agents", action="store_true")
    ap.add_argument("--mock", action="store_true", help="test the script itself with a fake model")
    ap.add_argument("--min-tps", type=float, default=4.0)
    ap.add_argument("--min-free-ram-mb", type=int, default=1200)
    ap.add_argument("--max-swap-growth-mb", type=int, default=256)
    ap.add_argument("--max-growth-mb", type=int, default=300)
    args = ap.parse_args()

    settings = load_settings()
    model = args.model or settings.backend.model
    ctx = args.ctx or int(settings.generation.num_ctx)

    candidates = [(model, ctx)]
    if args.auto_tune:
        settings._data["backend"]["model"], settings._data["generation"]["num_ctx"] = model, ctx
        candidates = OllamaClient(settings)._plan()

    installed = None if args.mock else OllamaClient(settings).list_models()
    if installed == [] and not args.mock:
        print("Cannot see any models. Is Ollama running? Try: ollama serve   (or: systemctl status ollama)")
        return 2

    RESULTS.mkdir(parents=True, exist_ok=True)
    chosen = None
    for cand_model, cand_ctx in candidates:
        if installed is not None and cand_model not in installed:
            print(f"\nSkipping {cand_model}: not installed. Install with: ollama pull {cand_model}")
            continue
        print(f"\nTesting {cand_model} at context {cand_ctx} ...")
        rep = run_once(settings, cand_model, cand_ctx, args, args.mock)
        print_report(rep)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = RESULTS / f"bench_{cand_model.replace(':', '-').replace('/', '-')}_{cand_ctx}_{stamp}.json"
        out.write_text(json.dumps(rep, indent=2))
        print(f" saved: {out}")
        if rep["verdict"]["stable"]:
            chosen = (cand_model, cand_ctx)
            break
        if not args.auto_tune:
            break
        print(" Unstable. Stepping down to the next setup.")

    if chosen:
        env = RESULTS / "recommended.env"
        env.write_text(f"LAI_BACKEND__MODEL={chosen[0]}\nLAI_GENERATION__NUM_CTX={chosen[1]}\n")
        print(f"\nStable setup: {chosen[0]} with context {chosen[1]}. Wrote {env}")
        print("To use it, copy those two lines into the .env file in the project folder.")
        return 0
    print("\nNo stable setup was found in this run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
