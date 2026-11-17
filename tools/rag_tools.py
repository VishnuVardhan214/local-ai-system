from __future__ import annotations

from tools.base import ToolContext, ToolError, tool


@tool("rag_search", "Search the user's uploaded documents. Returns the best matching passages with document names.",
      {"query": ("string", "Short search query with the key terms.", True),
       "k": ("integer", "Number of passages, 1 to 8. Default 4.", False)})
def rag_search(ctx: ToolContext, query: str, k: int = 4) -> str:
    k = max(1, min(8, k))
    hits = ctx.rag.search(query, k=k)
    if not hits:
        return "No relevant passages found in the uploaded documents."
    return "\n\n".join(f"[{h['doc']}, {h['locator']}] (score {h['score']})\n{h['text']}" for h in hits)


@tool("list_documents", "List the documents that have been uploaded and indexed.")
def list_documents(ctx: ToolContext) -> str:
    docs = ctx.rag.documents()
    if not docs:
        return "No documents have been uploaded yet."
    return "\n".join(f"{d['name']} ({d['n_chunks']} passages)" for d in docs)
