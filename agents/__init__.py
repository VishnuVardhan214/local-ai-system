"""Agents are discovered automatically: every module in this folder that uses
@register_agent adds an agent. See README, 'Adding a new agent'."""
import importlib
import pkgutil

from agents.base import AGENT_CLASSES, Agent, RunContext, register_agent  # noqa: F401

for _mod in pkgutil.iter_modules(__path__):
    if _mod.name != "base":
        importlib.import_module(f"agents.{_mod.name}")


def build_agents() -> dict[str, Agent]:
    return {name: cls() for name, cls in AGENT_CLASSES.items()}
