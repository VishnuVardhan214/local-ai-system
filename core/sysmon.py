"""RAM, swap, CPU and GPU readings. Shared by the web monitor and the benchmark."""
from __future__ import annotations

import shutil
import subprocess
import time

import psutil

_gpu_cache: tuple[float, dict | None] = (0.0, None)
MODEL_PROCESS_HINTS = ("ollama", "llama-server", "llama_server")


def read_gpu(max_age_s: float = 1.0) -> dict | None:
    """Returns the first NVIDIA GPU's memory and load, or None when there is no NVIDIA GPU."""
    global _gpu_cache
    now = time.time()
    if now - _gpu_cache[0] < max_age_s:
        return _gpu_cache[1]
    result = None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3).stdout.strip().splitlines()
            if out:
                name, used, total, util, driver = [x.strip() for x in out[0].split(",")]
                result = {"name": name, "vram_used_mb": int(used), "vram_total_mb": int(total),
                          "util_percent": int(util), "driver": driver}
        except (subprocess.SubprocessError, ValueError, OSError):
            result = None
    _gpu_cache = (now, result)
    return result


def model_process_rss_mb() -> float:
    """Resident memory of the model server processes. Includes memory-mapped model pages in RAM."""
    total = 0
    for p in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            name = (p.info["name"] or "").lower()
            cmd = " ".join(p.info["cmdline"] or []).lower()
            if any(h in name or h in cmd for h in MODEL_PROCESS_HINTS):
                total += p.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return round(total / 2**20, 1)


def snapshot() -> dict:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_mb": round((vm.total - vm.available) / 2**20),
        "ram_total_mb": round(vm.total / 2**20),
        "swap_used_mb": round(sw.used / 2**20),
        "swap_total_mb": round(sw.total / 2**20),
        "model_rss_mb": model_process_rss_mb(),
        "gpu": read_gpu(),
    }
