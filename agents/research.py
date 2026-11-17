"""Research agent: searches the local documents more than once and writes a structured report."""
from __future__ import annotations

from agents.base import Agent, format_sources, register_agent


@register_agent
class ResearchAgent(Agent):
    name = "research"
    title = "Research"
    description = "Looks through your local documents, compares findings and writes a structured report."
    keywords = ("research", "report", "summarize", "summarise", "summary", "compare", "overview",
                "findings", "analyze", "analyse", "investigate", "literature")
    tools = ("rag_search", "list_documents")
    max_steps = 4
    system_prompt = (
        "You are a careful research assistant working from the user's local documents. "
        "Some passages are already provided. If they are not enough, call rag_search with short, "
        "focused queries (at most 3 calls). Then write a report with exactly these sections:\n"
        "Summary\nKey findings\nEvidence\nGaps\n"
        "In Evidence, cite each source as [document name, location]. "
        "In Gaps, say what the documents do not cover. Never invent facts."
    )

    def run(self, run):
        tools = run.registry.specs(list(self.tools))
        hits = yield from self.retrieve(run, 6)
        prompt = run.user_text
        if hits:
            context, used = format_sources(hits, run.context_chars(0.45))
            run.sources = used
            yield {"type": "sources", "items": used}
            prompt = f"Passages already retrieved:\n{context}\n\nRequest: {run.user_text}"
        messages = run.build_messages(self, tools=tools, user_content=prompt)
        yield from self.loop(run, messages, tools)
