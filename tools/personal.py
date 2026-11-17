"""Notes, tasks and long-term facts for the personal assistant."""
from __future__ import annotations

from tools.base import ToolContext, ToolError, tool


@tool("add_note", "Save a note.",
      {"title": ("string", "Short title.", True),
       "text": ("string", "Note content.", True),
       "tags": ("string", "Comma separated tags. Optional.", False)})
def add_note(ctx: ToolContext, title: str, text: str, tags: str = "") -> str:
    nid = ctx.memory.add_note(title, text, tags)
    return f"Saved note #{nid}: {title}"


@tool("find_notes", "Search saved notes. An empty query lists the most recent ones.",
      {"query": ("string", "Words to look for. Optional.", False)})
def find_notes(ctx: ToolContext, query: str = "") -> str:
    rows = ctx.memory.search_notes(query)
    if not rows:
        return "No notes found."
    return "\n".join(f"#{r['id']} {r['title']}: {r['text'][:200]}" for r in rows)


@tool("add_task", "Add a task to the to-do list.",
      {"title": ("string", "What needs doing.", True),
       "due": ("string", "Due date or time in plain text. Optional.", False),
       "priority": ("string", "low, normal or high. Optional.", False)})
def add_task(ctx: ToolContext, title: str, due: str = "", priority: str = "normal") -> str:
    if priority not in ("low", "normal", "high"):
        priority = "normal"
    tid = ctx.memory.add_task(title, due, priority)
    return f"Added task #{tid}: {title}" + (f" (due {due})" if due else "")


@tool("list_tasks", "List tasks.",
      {"status": ("string", "open, done or all. Default open.", False)})
def list_tasks(ctx: ToolContext, status: str = "open") -> str:
    if status not in ("open", "done", "all"):
        status = "open"
    rows = ctx.memory.list_tasks(status)
    if not rows:
        return "No tasks."
    return "\n".join(f"#{r['id']} [{r['status']}] {r['title']}" + (f" due {r['due']}" if r["due"] else "")
                     + f" ({r['priority']})" for r in rows)


@tool("complete_task", "Mark a task as done.",
      {"task_id": ("integer", "The task number.", True)})
def complete_task(ctx: ToolContext, task_id: int) -> str:
    if not ctx.memory.complete_task(task_id):
        raise ToolError(f"No open task with number {task_id}.")
    return f"Task #{task_id} marked done."


@tool("remember_fact", "Store a lasting fact about the user or their preferences for future conversations.",
      {"key": ("string", "Short label such as 'favourite editor'.", True),
       "value": ("string", "The fact.", True)})
def remember_fact(ctx: ToolContext, key: str, value: str) -> str:
    ctx.memory.remember(key, value)
    return f"Remembered: {key} = {value}"


@tool("recall_facts", "Look up stored facts. An empty query lists recent ones.",
      {"query": ("string", "Words to look for. Optional.", False)})
def recall_facts(ctx: ToolContext, query: str = "") -> str:
    rows = ctx.memory.recall(query)
    return "\n".join(f"{r['key']}: {r['value']}" for r in rows) or "Nothing stored yet."
