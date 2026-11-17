"""File and command tools. Everything is confined to the configured workspace folders."""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from tools.base import ToolContext, ToolError, tool

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
MAX_READ_BYTES = 2_000_000
MAX_WRITE_CHARS = 200_000


def resolve_in_sandbox(ctx: ToolContext, raw: str) -> Path:
    """Turn a user or model supplied path into a real path inside an allowed folder."""
    roots = [r.resolve() for r in ctx.workspace_dirs]
    if not roots:
        raise ToolError("No workspace folder is configured.")
    candidate = Path(os.path.expanduser(raw or "."))
    if not candidate.is_absolute():
        candidate = roots[0] / candidate
    resolved = candidate.resolve()  # follows symlinks, so a link cannot point outside
    for root in roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    allowed = ", ".join(str(r) for r in roots)
    raise ToolError(f"Path is outside the allowed folders ({allowed}).")


def _is_text(path: Path) -> bool:
    with open(path, "rb") as fh:
        return b"\x00" not in fh.read(2048)


# ---- approval rules ---------------------------------------------------------
def _approve_write(args, ctx):
    try:
        target = resolve_in_sandbox(ctx, args.get("path", ""))
    except ToolError:
        return None  # the tool itself will refuse it
    return f"Overwrite existing file {target.name}" if target.exists() else None


def _approve_edit(args, ctx):
    return f"Modify file {args.get('path', '')}"


def _approve_delete(args, ctx):
    return f"Delete file {args.get('path', '')}"


def _approve_command(args, ctx):
    return f"Run command: {args.get('command', '')}"


def _approve_tests(args, ctx):
    if ctx.settings.security.get("auto_approve_tests", False):
        return None
    return "Run the test suite (this executes code in the workspace)"


# ---- tools ------------------------------------------------------------------
@tool("list_dir", "List files and folders in a workspace directory.",
      {"path": ("string", "Directory path, relative to the workspace. Default is the workspace root.", False)})
def list_dir(ctx: ToolContext, path: str = ".") -> str:
    target = resolve_in_sandbox(ctx, path)
    if not target.is_dir():
        raise ToolError("Not a directory.")
    rows = []
    for item in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if item.name in SKIP_DIRS:
            continue
        rows.append(f"{item.name}/" if item.is_dir() else f"{item.name}  ({item.stat().st_size} bytes)")
    return "\n".join(rows[:200]) or "(empty folder)"


@tool("read_file", "Read a text file from the workspace.",
      {"path": ("string", "File path relative to the workspace.", True),
       "max_chars": ("integer", "Maximum characters to return. Default 6000.", False)})
def read_file(ctx: ToolContext, path: str, max_chars: int = 6000) -> str:
    target = resolve_in_sandbox(ctx, path)
    if not target.is_file():
        raise ToolError("File does not exist.")
    if target.stat().st_size > MAX_READ_BYTES:
        raise ToolError("File is larger than 2 MB.")
    if not _is_text(target):
        raise ToolError("This looks like a binary file.")
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...[file continues, {len(text) - max_chars} more characters]"
    return text


@tool("search_files", "Search workspace text files for a regular expression. Returns file:line: text.",
      {"pattern": ("string", "Regular expression to look for.", True),
       "path": ("string", "Folder to search. Default is the workspace root.", False),
       "glob": ("string", "File name pattern such as *.py. Default is all files.", False)})
def search_files(ctx: ToolContext, pattern: str, path: str = ".", glob: str = "*") -> str:
    base = resolve_in_sandbox(ctx, path)
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise ToolError(f"Bad regular expression: {exc}")
    hits = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            fp = Path(root) / name
            if glob != "*" and not fp.match(glob):
                continue
            try:
                if fp.stat().st_size > 1_000_000 or not _is_text(fp):
                    continue
                for n, line in enumerate(fp.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{fp.relative_to(base)}:{n}: {line.strip()[:160]}")
                        if len(hits) >= 50:
                            return "\n".join(hits) + "\n...[stopped at 50 matches]"
            except OSError:
                continue
    return "\n".join(hits) or "No matches."


@tool("write_file", "Create a file, or replace it completely. Replacing an existing file needs the user's approval.",
      {"path": ("string", "File path relative to the workspace.", True),
       "content": ("string", "Full text content of the file.", True)},
      approval=_approve_write)
def write_file(ctx: ToolContext, path: str, content: str) -> str:
    target = resolve_in_sandbox(ctx, path)
    if len(content) > MAX_WRITE_CHARS:
        raise ToolError("Content is too large.")
    if target.is_dir():
        raise ToolError("That path is a directory.")
    existed = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"{'Replaced' if existed else 'Created'} {target.name} ({len(content)} characters)."


@tool("edit_file", "Replace one exact piece of text inside an existing file. Needs the user's approval.",
      {"path": ("string", "File path relative to the workspace.", True),
       "old_text": ("string", "Exact text to replace. It must appear exactly once.", True),
       "new_text": ("string", "Replacement text.", True)},
      approval=_approve_edit)
def edit_file(ctx: ToolContext, path: str, old_text: str, new_text: str) -> str:
    target = resolve_in_sandbox(ctx, path)
    if not target.is_file():
        raise ToolError("File does not exist.")
    text = target.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count == 0:
        raise ToolError("old_text was not found. Read the file again and copy the text exactly.")
    if count > 1:
        raise ToolError(f"old_text appears {count} times. Include more surrounding lines so it is unique.")
    target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {target.name}."


@tool("delete_file", "Delete one file from the workspace. Needs the user's approval.",
      {"path": ("string", "File path relative to the workspace.", True)},
      approval=_approve_delete)
def delete_file(ctx: ToolContext, path: str) -> str:
    target = resolve_in_sandbox(ctx, path)
    if not target.is_file():
        raise ToolError("Only single files can be deleted, and this one does not exist.")
    target.unlink()
    return f"Deleted {target.name}."


def _run_process(argv: list[str], cwd: Path, timeout: int) -> str:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(cwd), "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        proc = subprocess.run(argv, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ToolError(f"Timed out after {timeout} seconds.")
    except FileNotFoundError:
        raise ToolError(f"Command not found: {argv[0]}")
    out = (proc.stdout or "") + (proc.stderr or "")
    return f"exit code {proc.returncode}\n{out.strip()}"


@tool("run_tests", "Run pytest inside a workspace folder and return the result. Needs the user's approval.",
      {"path": ("string", "Folder to run in, relative to the workspace. Default is the root.", False),
       "args": ("string", "Extra pytest arguments such as -k name. Optional.", False)},
      approval=_approve_tests)
def run_tests(ctx: ToolContext, path: str = ".", args: str = "") -> str:
    cwd = resolve_in_sandbox(ctx, path)
    if not cwd.is_dir():
        raise ToolError("Not a directory.")
    extra = shlex.split(args) if args else []
    if any(a.startswith(("--rootdir", "-p", "--confcutdir")) for a in extra):
        raise ToolError("That pytest option is not allowed.")
    return _run_process([sys.executable, "-m", "pytest", "-x", "-q", *extra], cwd, timeout=120)


@tool("run_command", "Run one command in the workspace without a shell. Always needs the user's approval.",
      {"command": ("string", "The command, for example: python script.py", True)},
      approval=_approve_command)
def run_command(ctx: ToolContext, command: str) -> str:
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ToolError(f"Could not parse the command: {exc}")
    if not argv:
        raise ToolError("Empty command.")
    blocked = set(ctx.settings.security.blocked_commands)
    if Path(argv[0]).name in blocked:
        raise ToolError(f"The command {argv[0]} is blocked by the security settings.")
    cwd = resolve_in_sandbox(ctx, ".")
    return _run_process(argv, cwd, timeout=60)
