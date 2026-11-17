"""Approval gate for risky tool calls.

An agent asks for approval, the web page shows Approve and Deny buttons, and the
agent waits. If nobody answers before the timeout the action is denied.
"""
from __future__ import annotations

import threading
import uuid


class ApprovalManager:
    def __init__(self, timeout_s: float = 120.0, auto: str | None = None):
        self.timeout_s = timeout_s
        self.auto = auto  # None, "approve" or "deny". Used by tests and scripts only.
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}

    def create(self, tool: str, args: dict, reason: str) -> dict:
        req = {"id": uuid.uuid4().hex[:10], "tool": tool, "args": args, "reason": reason}
        with self._lock:
            self._pending[req["id"]] = {"event": threading.Event(), "approved": False, "req": req}
        return req

    def resolve(self, approval_id: str, approved: bool) -> bool:
        with self._lock:
            entry = self._pending.get(approval_id)
        if not entry:
            return False
        entry["approved"] = bool(approved)
        entry["event"].set()
        return True

    def wait(self, approval_id: str) -> bool:
        with self._lock:
            entry = self._pending.get(approval_id)
        if not entry:
            return False
        if self.auto is not None:
            result = self.auto == "approve"
        else:
            entry["event"].wait(self.timeout_s)
            result = entry["approved"] if entry["event"].is_set() else False
        with self._lock:
            self._pending.pop(approval_id, None)
        return result

    def pending(self) -> list[dict]:
        with self._lock:
            return [e["req"] for e in self._pending.values()]
