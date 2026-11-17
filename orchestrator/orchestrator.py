"""Central orchestrator: routes a request, runs the agent, tracks state, retries, saves history."""
from __future__ import annotations

import logging
import threading
import time
from typing import Iterator

from agents import RunContext, build_agents
from core.llm import LLMError
from orchestrator.router import Router
from tools import REGISTRY, ToolContext

log = logging.getLogger("orchestrator")


class Orchestrator:
    def __init__(self, settings, llm, memory, rag, approvals):
        self.s = settings
        self.llm = llm
        self.memory = memory
        self.rag = rag
        self.approvals = approvals
        self.agents = build_agents()
        self.router = Router(self.agents, llm)
        self.tool_ctx = ToolContext(settings=settings, memory=memory, rag=rag, workspace_dirs=settings.workspace_dirs)
        # One request at a time keeps a single model in memory and stops requests fighting for it.
        self.gate = threading.Lock()
        self.state: dict = {"status": "idle", "agent": None, "session_id": None, "step": "", "since": time.time()}

    # ---- state ----------------------------------------------------------------
    def _set_state(self, **kw) -> None:
        self.state.update(kw, since=time.time())

    def agent_list(self) -> list[dict]:
        return [{"name": a.name, "title": a.title, "description": a.description, "tools": list(a.tools)}
                for a in self.agents.values()]

    # ---- main entry ------------------------------------------------------------
    def handle(self, session_id: str | None, text: str, agent: str = "auto") -> Iterator[dict]:
        if not session_id or not self.memory.conversation_exists(session_id):
            session_id = self.memory.create_conversation(text[:50])
        elif not self.memory.get_title(session_id):
            self.memory.rename_conversation(session_id, text[:50])
        yield {"type": "session", "id": session_id}

        if not self.gate.acquire(blocking=False):
            yield {"type": "status", "text": "Another request is running. Waiting for the model to be free."}
            self.gate.acquire()
        try:
            yield from self._run(session_id, text, agent)
        finally:
            self._set_state(status="idle", agent=None, session_id=None, step="")
            self.gate.release()
        self._maybe_summarize(session_id)

    def _run(self, session_id: str, text: str, forced: str) -> Iterator[dict]:
        self._set_state(status="routing", session_id=session_id, step="choosing an agent")
        name, reason = self.router.route(text, forced)
        agent = self.agents[name]
        yield {"type": "route", "agent": name, "title": agent.title, "reason": reason}

        summary, history = self.memory.prompt_history(session_id)
        run = RunContext(settings=self.s, llm=self.llm, memory=self.memory, rag=self.rag, tool_ctx=self.tool_ctx,
                         approvals=self.approvals, registry=REGISTRY, session_id=session_id, user_text=text,
                         summary=summary, history=[{"role": m["role"], "content": m["content"]} for m in history])
        self.memory.add_message(session_id, "user", text)

        retries = int(self.s.agents.get("retries", 1))
        started = time.time()
        for attempt in range(retries + 1):
            progressed = False
            try:
                self._set_state(status="running", agent=name, step="thinking")
                for ev in agent.run(run):
                    if ev["type"] in ("token", "tool_call", "approval_request"):
                        progressed = True
                    if ev["type"] == "tool_call":
                        self._set_state(step=f"using {ev['name']}")
                    elif ev["type"] == "token":
                        self._set_state(step="writing the answer")
                    yield ev
                break
            except LLMError as exc:
                log.warning("model error (attempt %d): %s", attempt + 1, exc)
                if progressed or attempt >= retries:
                    yield {"type": "error", "message": str(exc)}
                    break
                yield {"type": "status", "text": f"Model error, trying again ({attempt + 1}/{retries})."}
                run.tool_log.clear()
                run.final_text = ""
            except Exception as exc:  # keep the server alive whatever an agent does
                log.exception("agent crashed")
                yield {"type": "error", "message": f"Internal error in the {name} agent: {type(exc).__name__}"}
                break

        meta = {"tools": run.tool_log, "sources": run.sources, "stats": run.totals}
        self.memory.add_message(session_id, "assistant", run.final_text, agent=name, meta=meta)
        yield {"type": "done", "agent": name, "session_id": session_id,
               "seconds": round(time.time() - started, 2), "tokens": run.totals["tokens"]}

    # ---- keeping history small ---------------------------------------------------
    def _maybe_summarize(self, session_id: str) -> None:
        m = self.s.memory
        old = self.memory.messages_to_fold(session_id, int(m.keep_last_messages), int(m.summarize_after_chars))
        if not old:
            return
        previous, _ = self.memory.prompt_history(session_id)
        transcript = "\n".join(f"{x['role']}: {x['content'][:600]}" for x in old)
        prompt = (f"Earlier summary:\n{previous or '(none)'}\n\nNew messages:\n{transcript}\n\n"
                  "Write an updated summary in under 150 words. Keep names, decisions, file names and open tasks.")
        with self.gate:
            try:
                summary = self.llm.complete(prompt, system="You write short, factual conversation summaries.",
                                            num_predict=300)
            except Exception:
                log.warning("summary failed, will try again later")
                return
        if summary.strip():
            self.memory.set_summary(session_id, summary.strip(), old[-1]["id"])
