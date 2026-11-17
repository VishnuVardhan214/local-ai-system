"""Phase 1: look at the real machine before changing anything.

Run:  python scripts/inspect_hardware.py
It only reads information. It installs nothing and changes nothing.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(cmd: list[str], timeout: int = 8) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or out.stderr or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def gb(n: float) -> str:
    return f"{n / 2**30:.1f} GB"


def section(title: str) -> None:
    print(f"\n== {title} ==")


def cpu_name() -> str:
    if platform.system() == "Linux":
        m = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text(errors="ignore"))
        if m:
            return m.group(1).strip()
    return platform.processor() or "unknown"


def main() -> int:
    notes: list[str] = []

    section("Operating system")
    print(f"system   {platform.system()} {platform.release()}")
    print(f"detail   {platform.platform()}")
    if platform.system() == "Linux":
        osr = Path("/etc/os-release")
        if osr.exists():
            m = re.search(r'PRETTY_NAME="?([^"\n]+)', osr.read_text())
            print(f"distro   {m.group(1) if m else 'unknown'}")
        if "microsoft" in platform.release().lower():
            notes.append("This looks like WSL. GPU use under WSL needs the Windows NVIDIA driver with WSL support.")

    section("Memory")
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    print(f"RAM      {gb(vm.total)} total, {gb(vm.available)} available")
    print(f"swap     {gb(sw.total)} total, {gb(sw.used)} used")
    if vm.total < 12 * 2**30:
        notes.append("Less than 12 GB of RAM is visible. Close other programs before running larger models.")
    if sw.total == 0:
        notes.append("There is no swap. A model that does not fit will be killed by the system instead of slowing down.")

    section("CPU")
    print(f"model    {cpu_name()}")
    print(f"cores    {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} threads")
    phys = psutil.cpu_count(logical=False) or 4
    print(f"suggest  num_thread {max(2, phys - 2)} (leaves two cores for the system)")

    section("GPU")
    smi = shutil.which("nvidia-smi")
    if smi:
        info = run([smi, "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"])
        print(f"nvidia   {info or 'nvidia-smi found but gave no output'}")
        header = run([smi])
        m = re.search(r"CUDA Version:\s*([\d.]+)", header)
        print(f"CUDA     driver supports up to {m.group(1)}" if m else "CUDA     version not reported")
        procs = run([smi, "--query-compute-apps=name,used_memory", "--format=csv,noheader"])
        print(f"in use   {procs or 'no compute programs are using the GPU'}")
    else:
        print("nvidia   nvidia-smi not found")
        notes.append("No NVIDIA driver found. Ollama will run on the CPU only. On Ubuntu try: ubuntu-drivers devices")
    nvcc = shutil.which("nvcc")
    print(f"nvcc     {run([nvcc, '--version']).splitlines()[-1] if nvcc else 'not installed (not needed for Ollama)'}")

    section("Disk")
    target = Path(os.environ.get("OLLAMA_MODELS", Path.home()))
    target = target if target.exists() else Path.home()
    du = shutil.disk_usage(target)
    print(f"checked  {target}")
    print(f"free     {gb(du.free)} of {gb(du.total)}")
    if du.free < 15 * 2**30:
        notes.append("Less than 15 GB free. Model files are several GB each.")

    section("Ollama")
    ollama = shutil.which("ollama")
    if ollama:
        print(f"binary   {ollama}")
        print(f"version  {run([ollama, '--version']).splitlines()[-1]}")
        print("installed models:")
        print("  " + (run([ollama, "list"]).replace("\n", "\n  ") or "(none, or the server is not running)"))
        print("loaded right now:")
        print("  " + (run([ollama, "ps"]).replace("\n", "\n  ") or "(nothing)"))
        if platform.system() == "Linux" and shutil.which("systemctl"):
            print(f"service  {run(['systemctl', 'is-active', 'ollama']) or 'unknown'}")
            env = run(["systemctl", "show", "ollama", "-p", "Environment"])
            print(f"env      {env or 'none shown'}")
    else:
        print("Ollama is not installed. Run scripts/01_setup_ollama.sh")

    section("Python environments")
    print(f"this python   {sys.version.split()[0]} at {sys.executable}")
    print(f"in a venv     {sys.prefix != getattr(sys, 'base_prefix', sys.prefix)}")
    for name in ("venv", ".venv"):
        if (ROOT / name).exists():
            print(f"project env   {ROOT / name}")
    if shutil.which("conda"):
        print("conda envs:\n  " + run(["conda", "env", "list"]).replace("\n", "\n  "))
    missing = []
    for mod in ("fastapi", "uvicorn", "httpx", "yaml", "numpy", "psutil", "pypdf", "docx", "multipart"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    print("packages      all present" if not missing else f"packages      missing: {', '.join(missing)}")

    if notes:
        section("Things to know")
        for n in notes:
            print("-", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
