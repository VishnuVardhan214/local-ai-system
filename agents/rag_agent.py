"""RAG agent: always retrieves first, then answers only from the passages and cites them."""
from __future__ import annotations

from agents.base import Agent, format_sources, register_agent


@register_agent
class RagAgent(Agent):
    name = "rag"
    title = "Document Q&A"
    description = "Answers questions from your uploaded documents and cites the passages it used."
    keywords = ("document", "documents", "pdf", "docx", "uploaded", "according to", "in the file",
                "in my files", "in the paper", "source", "cite")
    tools = ()
    system_prompt = (
        "You answer questions using only the numbered sources given in the user message. "
        "Cite the sources you use like [1] or [2]. "
        "If the sources do not contain the answer, say clearly that the documents do not cover it. "
        "Do not use outside knowledge. Keep the answer short and direct."
    )

    def run(self, run):
        hits = yield from self.retrieve(run, int(run.settings.rag.top_k))
        if not hits:
            if not run.rag.documents():
                run.final_text = "No documents are indexed yet. Upload a PDF, TXT, Markdown or DOCX file first."
            else:
                run.final_text = "I could not find anything relevant to that in the uploaded documents."
            yield {"type": "token", "text": run.final_text}
            return
        context, used = format_sources(hits, run.context_chars(0.5))
        run.sources = used
        yield {"type": "sources", "items": used}
        prompt = f"Sources:\n{context}\n\nQuestion: {run.user_text}"
        messages = run.build_messages(self, tools=None, user_content=prompt)
        text, _ = yield from self.stream_step(run, messages, None)
        run.final_text = text
