"""Personal assistant: tasks, notes and long-term facts. Also handles general chat."""
from __future__ import annotations

from agents.base import Agent, register_agent


@register_agent
class PersonalAgent(Agent):
    name = "personal"
    title = "Personal assistant"
    description = "Manages tasks and notes, remembers facts about you, and handles everyday questions."
    keywords = ("task", "todo", "to-do", "remind", "reminder", "note", "notes", "schedule", "remember",
                "deadline", "checklist", "plan my")
    tools = ("add_task", "list_tasks", "complete_task", "add_note", "find_notes", "remember_fact", "recall_facts")
    use_facts = True
    max_steps = 4
    system_prompt = (
        "You are a friendly personal assistant. "
        "When the user asks you to add a task, save a note or remember something, use the matching tool, "
        "then confirm in one short sentence. "
        "Use list_tasks or find_notes when they ask what is on their list. "
        "For everyday questions, answer briefly and plainly without using tools."
    )
