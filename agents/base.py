"""Agent base class and the shared tool-calling loop.

An agent is a small Python object: a system prompt, a list of allowed tools and a
loop. All agents share ONE model through run.llm, so adding agents costs no memory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

from tools.base import ToolError

AGENT_CLASSES: dict[str, type] = {}


def register_agent(cls):
    AGENT_CLASSES[cls.name] = cls
    return cls


def est_tokens(text: str) -> int:
    """Rough token estimate. Slightly pessimistic on purpose."""
    return len(text) // 3 + 4


@dataclass
class RunContext:
    settings: Any
    llm: Any
    memory: Any
    rag: Any
    tool_ctx: Any
    approvals: Any
    registry: Any
    session_id: str
    user_text: str
    summary: str = ""
    history: list[dict] = field(default_factory=list)
    final_text: str = ""
    tool_log: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    totals: dict = field(default_factory=lambda: {"tokens": 0, "seconds": 0.0, "tokens_per_s": 0.0})

    def add_stats(self, stats: dict) -> None:
        self.totals["tokens"] += stats.get("eval_count", 0)
        self.totals["seconds"] += stats.get("total_s", 0.0)
        if stats.get("tokens_per_s"):
            self.totals["tokens_per_s"] = stats["tokens_per_s"]

    def prompt_budget(self, tools: list | None) -> int:
        """Tokens available for the prompt after reserving room for the reply and tool schemas."""
        reserve = int(self.settings.generation.get("max_tokens", 1024))
        tool_tokens = est_tokens(json.dumps(tools)) if tools else 0
        return int(self.llm.num_ctx) - reserve - tool_tokens

    def build_messages(self, agent: "Agent", tools: list | None = None, user_content: str | None = None) -> list[dict]:
        system = {"role": "system", "content": agent.build_system(self)}
        user = {"role": "user", "content": user_content if user_content is not None else self.user_text}
        budget = self.prompt_budget(tools)
        used = est_tokens(system["content"]) + est_tokens(user["content"])
        kept: list[dict] = []
        for m in reversed(self.history):  # newest first, stop when the window is full
            cost = est_tokens(m["content"]) + 4
            if used + cost > budget:
                break
            kept.append({"role": m["role"], "content": m["content"]})
            used += cost
        kept.reverse()
        return [system, *kept, user]

    def context_chars(self, share: float = 0.5) -> int:
        """Characters of retrieved text that fit in the window (share of the prompt budget)."""
        cap = int(self.settings.rag.get("max_context_chars", 4000))
        return min(cap, int(self.prompt_budget(None) * share * 3))


def format_sources(hits: list[dict], max_chars: int) -> tuple[str, list[dict]]:
    parts, used, kept = [], 0, []
    for n, h in enumerate(hits, start=1):
        text = h["text"]
        room = max_chars - used
        if room < 200:
            break
        text = text[:room]
        parts.append(f"[{n}] ({h['doc']}, {h['locator']})\n{text}")
        used += len(text) + 40
        kept.append({"n": n, "doc": h["doc"], "locator": h["locator"], "score": h["score"],
                     "snippet": h["text"][:160].replace("\n", " ")})
    return "\n\n".join(parts), kept


class Agent:
    name = "base"
    title = "Base agent"
    description = ""
    keywords: tuple = ()
    tools: tuple = ()
    use_facts = False
    max_steps: int | None = None
    system_prompt = "You are a helpful assistant."

    # ---- prompt -----------------------------------------------------------
    def build_system(self, run: RunContext) -> str:
        parts = [self.system_prompt]
        if self.use_facts:
            facts = run.memory.recall("", limit=int(run.settings.memory.get("max_facts_in_prompt", 12)))
            if facts:
                parts.append("Known facts about the user:\n" + "\n".join(f"- {f['key']}: {f['value']}" for f in facts))
        if run.summary:
            parts.append("Summary of the earlier conversation:\n" + run.summary[:1500])
        return "\n\n".join(parts)

    # ---- one model call ---------------------------------------------------
    def stream_step(self, run: RunContext, messages: list[dict], tools: list | None):
        text, calls = [], []
        for ev in run.llm.stream_chat(messages, tools=tools):
            kind = ev["type"]
            if kind == "token":
                text.append(ev["text"])
                yield {"type": "token", "text": ev["text"]}
            elif kind == "tool_call":
                calls.append({"name": ev["name"], "arguments": ev["arguments"]})
            elif kind == "fallback":
                yield {"type": "status", "text": f"Model {ev['model']} did not fit ({ev['reason']}). Trying a smaller setup."}
            elif kind == "done":
                run.add_stats(ev["stats"])
                yield {"type": "stats", **ev["stats"]}
        return "".join(text), calls

    # ---- tool execution -----------------------------------------------------
    def execute_tool(self, run: RunContext, call: dict):
        name, args = call["name"], call["arguments"]
        spec = run.registry.get(name)
        if spec is None or name not in self.tools:
            return f"ERROR: unknown tool '{name}'. Available tools: {', '.join(self.tools) or 'none'}."
        try:
            clean = run.registry.clean_args(spec, args)
        except ToolError as exc:
            return f"ERROR: {exc}"
        reason = run.registry.approval_reason(spec, clean, run.tool_ctx)
        if reason:
            req = run.approvals.create(name, clean, reason)
            yield {"type": "approval_request", **req}
            approved = run.approvals.wait(req["id"])
            yield {"type": "approval_result", "id": req["id"], "approved": approved}
            if not approved:
                return ("ERROR: The user did not approve this action. Do not try it again. "
                        "Tell the user what you wanted to do and why.")
        return run.registry.run(spec, clean, run.tool_ctx)

    # ---- the loop ---------------------------------------------------------------
    def loop(self, run: RunContext, messages: list[dict], tools: list | None) -> Iterator[dict]:
        limit = self.max_steps or int(run.settings.agents.get("max_steps", 6))
        seen: set[str] = set()
        for _ in range(limit):
            text, calls = yield from self.stream_step(run, messages, tools)
            if not calls:
                run.final_text = text
                return
            messages.append({"role": "assistant", "content": text,
                             "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
            for call in calls:
                yield {"type": "tool_call", "name": call["name"], "args": call["arguments"]}
                key = call["name"] + json.dumps(call["arguments"], sort_keys=True, default=str)
                if key in seen:
                    result = "ERROR: You already made this exact call. Use the earlier result or answer now."
                else:
                    seen.add(key)
                    result = yield from self.execute_tool(run, call)
                ok = not result.startswith("ERROR")
                run.tool_log.append({"name": call["name"], "args": call["arguments"], "ok": ok, "preview": result[:300]})
                yield {"type": "tool_result", "name": call["name"], "ok": ok, "preview": result[:300]}
                messages.append({"role": "tool", "tool_name": call["name"], "content": result})
        # Out of steps: ask for a final answer with no tools.
        messages.append({"role": "user", "content": "Stop using tools now and give your final answer."})
        text, _ = yield from self.stream_step(run, messages, None)
        run.final_text = text

    def run(self, run: RunContext) -> Iterator[dict]:
        tools = run.registry.specs(list(self.tools)) or None
        messages = run.build_messages(self, tools=tools)
        yield from self.loop(run, messages, tools)

    # ---- shared helper: retrieve before asking the model -----------------------------
    def retrieve(self, run: RunContext, k: int):
        yield {"type": "tool_call", "name": "rag_search", "args": {"query": run.user_text, "k": k}}
        try:
            hits = run.rag.search(run.user_text, k=k)
        except Exception as exc:
            msg = f"ERROR: search failed: {exc}"
            run.tool_log.append({"name": "rag_search", "args": {"query": run.user_text}, "ok": False, "preview": msg})
            yield {"type": "tool_result", "name": "rag_search", "ok": False, "preview": msg}
            return []
        preview = f"{len(hits)} passages found" if hits else "no relevant passages"
        run.tool_log.append({"name": "rag_search", "args": {"query": run.user_text}, "ok": True, "preview": preview})
        yield {"type": "tool_result", "name": "rag_search", "ok": True, "preview": preview}
        return hits
