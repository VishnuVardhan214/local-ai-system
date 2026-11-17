"""Decides which agent handles a request.

Keyword rules come first because they are instant and cost no model time. Only a
request that matches nothing is sent to the model for a one-word classification.
Agents supply their own keywords, so a new agent joins the routing automatically.
"""
from __future__ import annotations

import re

PRIORITY = {"coding": 0, "research": 1, "rag": 2, "personal": 3}  # used to break ties
CODE_FILE = re.compile(r"\b[\w./-]+\.(py|js|ts|java|cpp|c|h|rs|go|sh|sql|html|css)\b")


class Router:
    DEFAULT = "personal"

    def __init__(self, agents: dict, llm):
        self.agents = agents
        self.llm = llm

    def _scores(self, text: str) -> dict[str, list[str]]:
        low = text.lower()
        found: dict[str, list[str]] = {}
        for name, agent in self.agents.items():
            hits = [k for k in agent.keywords if re.search(rf"\b{re.escape(k)}\b", low)]
            if name == "coding" and CODE_FILE.search(low):
                hits.append("code file name")
            if hits:
                found[name] = hits
        return found

    def route(self, text: str, forced: str | None = None) -> tuple[str, str]:
        if forced and forced != "auto":
            if forced in self.agents:
                return forced, "chosen by you"
            return self.DEFAULT, f"unknown agent '{forced}', using {self.DEFAULT}"
        found = self._scores(text)
        if found:
            best = sorted(found, key=lambda n: (-len(found[n]), PRIORITY.get(n, 10)))[0]
            return best, "keywords: " + ", ".join(found[best][:4])
        return self._ask_model(text)

    def _ask_model(self, text: str) -> tuple[str, str]:
        labels = ", ".join(self.agents)
        desc = "\n".join(f"{n}: {a.description}" for n, a in self.agents.items())
        system = (f"Pick the best agent for the user's request.\n{desc}\n"
                  f"Reply with only one label from: {labels}.")
        try:
            answer = self.llm.complete(text[:500], system=system, num_predict=8).lower()
        except Exception:
            return self.DEFAULT, "model unavailable, using default"
        for name in self.agents:
            if re.search(rf"\b{re.escape(name)}\b", answer):
                return name, "chosen by the model"
        return self.DEFAULT, "no clear match, using default"
