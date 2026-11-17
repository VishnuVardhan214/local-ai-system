"""Tool registry. A tool is a plain function plus a description the model can read."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class ToolError(Exception):
    """An expected failure. The message is shown to the model so it can recover."""


@dataclass
class ToolContext:
    settings: Any
    memory: Any
    rag: Any
    workspace_dirs: list[Path] = field(default_factory=list)


@dataclass
class ToolSpec:
    name: str
    description: str
    params: dict          # name -> (json type, description, required)
    fn: Callable
    approval: Callable | None = None   # (args, ctx) -> reason string if approval is needed

    def schema(self) -> dict:
        props = {k: {"type": t, "description": d} for k, (t, d, _) in self.params.items()}
        required = [k for k, (_, _, req) in self.params.items() if req]
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": props, "required": required}}}


_CASTS = {"string": str, "integer": int, "number": float, "boolean": lambda v: v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")}


class ToolRegistry:
    MAX_OUTPUT = 4000

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, names: list[str]) -> list[dict]:
        return [self._tools[n].schema() for n in names if n in self._tools]

    def clean_args(self, spec: ToolSpec, args: dict) -> dict:
        if not isinstance(args, dict):
            raise ToolError("Arguments must be an object.")
        out = {}
        for key, (jtype, _, required) in spec.params.items():
            if key not in args or args[key] is None:
                if required:
                    raise ToolError(f"Missing required argument: {key}")
                continue
            try:
                out[key] = _CASTS[jtype](args[key])
            except (ValueError, TypeError):
                raise ToolError(f"Argument {key} must be of type {jtype}.")
        return out  # unknown arguments are dropped on purpose

    def approval_reason(self, spec: ToolSpec, args: dict, ctx: ToolContext) -> str | None:
        return spec.approval(args, ctx) if spec.approval else None

    def run(self, spec: ToolSpec, args: dict, ctx: ToolContext) -> str:
        try:
            result = spec.fn(ctx, **self.clean_args(spec, args))
        except ToolError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # a tool bug must not crash the agent loop
            return f"ERROR: {type(exc).__name__}: {exc}"
        text = result if isinstance(result, str) else str(result)
        if len(text) > self.MAX_OUTPUT:
            text = text[: self.MAX_OUTPUT] + f"\n...[truncated, {len(text) - self.MAX_OUTPUT} more characters]"
        return text


REGISTRY = ToolRegistry()


def tool(name: str, description: str, params: dict | None = None, approval: Callable | None = None):
    """Decorator that registers a function as a tool.

    params maps argument name to (json type, description, required).
    approval is an optional function (args, ctx) that returns a reason string when
    the call needs the user's approval, or None when it can run directly.
    """
    def wrap(fn: Callable) -> Callable:
        REGISTRY.register(ToolSpec(name, description, params or {}, fn, approval))
        return fn
    return wrap
