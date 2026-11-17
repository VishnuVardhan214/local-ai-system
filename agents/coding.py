"""Coding agent: reads project files, edits them, runs tests. Risky steps need approval."""
from __future__ import annotations

from agents.base import Agent, register_agent


@register_agent
class CodingAgent(Agent):
    name = "coding"
    title = "Coding"
    description = "Reads your project files, writes and edits code, runs tests and explains the changes."
    keywords = ("code", "function", "class", "bug", "error", "traceback", "refactor", "unit test", "pytest",
                "script", "python", "javascript", "compile", "implement", "repo", "stack trace")
    tools = ("list_dir", "read_file", "search_files", "write_file", "edit_file", "delete_file", "run_tests", "run_command")
    max_steps = 8
    system_prompt = (
        "You are a careful coding assistant. You can only work inside the workspace folder. "
        "Look at the files with list_dir, read_file and search_files before changing anything. "
        "Prefer edit_file for small changes and write_file for new files. "
        "Overwriting, editing and deleting files, running tests and running commands all need the user's "
        "approval, so make each such call only when it is really needed. "
        "After changing code, run the tests if the project has them. "
        "Finish with a short explanation: which files you changed and why."
    )

    def build_system(self, run):
        base = super().build_system(run)
        roots = ", ".join(str(p) for p in run.tool_ctx.workspace_dirs)
        return f"{base}\n\nWorkspace folder: {roots}. Relative paths start there."
