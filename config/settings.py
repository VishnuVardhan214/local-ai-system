"""Loads config/config.yaml, then applies .env and LAI_* environment overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def _load_dotenv(path: Path) -> None:
    """Tiny .env reader. Real environment variables win over the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _apply_env(data: dict) -> None:
    for key, raw in os.environ.items():
        if not key.startswith("LAI_") or "__" not in key:
            continue
        parts = key[4:].lower().split("__")
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                break
        else:
            try:
                node[parts[-1]] = yaml.safe_load(raw)
            except yaml.YAMLError:
                node[parts[-1]] = raw


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


class Section:
    """Read-only attribute access over a nested dict."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Section(value) if isinstance(value, dict) else value

    def get(self, name: str, default: Any = None) -> Any:
        value = self._data.get(name, default)
        return Section(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        return dict(self._data)


class Settings(Section):
    def __init__(self, data: dict):
        super().__init__(data)

    @property
    def data_dir(self) -> Path:
        return expand_path(self._data["paths"]["data_dir"])

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def workspace_dirs(self) -> list[Path]:
        return [expand_path(p) for p in self._data["paths"]["workspace_dirs"]]

    @property
    def memory_db(self) -> Path:
        return self.data_dir / "memory.db"

    @property
    def rag_db(self) -> Path:
        return self.data_dir / "rag.db"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    def ensure_dirs(self) -> None:
        for p in [self.data_dir, self.docs_dir, self.log_dir, *self.workspace_dirs]:
            p.mkdir(parents=True, exist_ok=True)


def load_settings(config_path: str | Path | None = None, overrides: dict | None = None) -> Settings:
    _load_dotenv(ROOT / ".env")
    path = Path(config_path or os.environ.get("LAI_CONFIG", DEFAULT_CONFIG))
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _apply_env(data)
    for dotted, value in (overrides or {}).items():
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return Settings(data)
