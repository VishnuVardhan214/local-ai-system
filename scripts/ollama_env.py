"""Turns the ollama_server block of config/config.yaml into settings for the Ollama SERVER.

The Ollama server reads these from its own environment, not from the Python app,
so they must be applied to the service. Examples:

    python scripts/ollama_env.py --print       # shell export lines
    python scripts/ollama_env.py --systemd     # Linux service drop-in (run with sudo)
    python scripts/ollama_env.py --windows     # setx commands for Windows
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import expand_path, load_settings  # noqa: E402

DROPIN = Path("/etc/systemd/system/ollama.service.d/local-ai.conf")


def server_env() -> dict[str, str]:
    raw = load_settings().ollama_server.to_dict()
    env = {k: str(v) for k, v in raw.items()}
    if "OLLAMA_MODELS" in env:
        env["OLLAMA_MODELS"] = str(expand_path(env["OLLAMA_MODELS"]))
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print", action="store_true")
    mode.add_argument("--systemd", action="store_true")
    mode.add_argument("--windows", action="store_true")
    args = ap.parse_args()
    env = server_env()

    if args.print:
        for k, v in env.items():
            print(f'export {k}="{v}"')
        return 0
    if args.windows:
        for k, v in env.items():
            print(f'setx {k} "{v}"')
        print("Then quit Ollama from the tray icon and start it again.")
        return 0

    if os.geteuid() != 0:
        print("Run this with sudo so it can write the service file.")
        return 1
    if not shutil.which("systemctl") or not Path("/run/systemd/system").exists():
        print("This machine is not running systemd, so there is no service to configure.")
        print("Use --print and paste the lines into your shell before running: ollama serve")
        return 1
    lines = ["[Service]"] + [f'Environment="{k}={v}"' for k, v in env.items()]
    DROPIN.parent.mkdir(parents=True, exist_ok=True)
    DROPIN.write_text("\n".join(lines) + "\n")
    if "OLLAMA_MODELS" in env:
        Path(env["OLLAMA_MODELS"]).mkdir(parents=True, exist_ok=True)
        subprocess.run(["chown", "-R", "ollama:ollama", env["OLLAMA_MODELS"]], check=False)
    try:
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "restart", "ollama"], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"The settings file was written to {DROPIN}, but systemctl failed: {exc}")
        print("Check the service with: systemctl status ollama")
        return 1
    print(f"Wrote {DROPIN} and restarted Ollama.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
