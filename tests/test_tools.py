import os

import pytest

from tools import REGISTRY, ToolError
from tools.fs import resolve_in_sandbox


def call(ctx, name, **args):
    spec = REGISTRY.get(name)
    return REGISTRY.run(spec, args, ctx)


def test_relative_path_stays_inside(tool_ctx):
    p = resolve_in_sandbox(tool_ctx, "a/b.txt")
    assert str(p).startswith(str(tool_ctx.workspace_dirs[0]))


@pytest.mark.parametrize("bad", ["../secret.txt", "/etc/passwd", "a/../../x", "~/.ssh/id_rsa"])
def test_escape_attempts_are_refused(tool_ctx, bad):
    with pytest.raises(ToolError):
        resolve_in_sandbox(tool_ctx, bad)


def test_symlink_pointing_outside_is_refused(tool_ctx, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = tool_ctx.workspace_dirs[0] / "link.txt"
    os.symlink(outside, link)
    assert call(tool_ctx, "read_file", path="link.txt").startswith("ERROR")


def test_write_then_read(tool_ctx):
    assert "Created" in call(tool_ctx, "write_file", path="notes/a.txt", content="hello")
    assert call(tool_ctx, "read_file", path="notes/a.txt") == "hello"


def test_overwrite_needs_approval_but_new_file_does_not(tool_ctx):
    spec = REGISTRY.get("write_file")
    assert REGISTRY.approval_reason(spec, {"path": "new.txt", "content": "x"}, tool_ctx) is None
    call(tool_ctx, "write_file", path="new.txt", content="x")
    assert REGISTRY.approval_reason(spec, {"path": "new.txt", "content": "y"}, tool_ctx)


def test_delete_edit_and_command_always_need_approval(tool_ctx):
    for name, args in [("delete_file", {"path": "x"}), ("edit_file", {"path": "x", "old_text": "a", "new_text": "b"}),
                       ("run_command", {"command": "echo hi"})]:
        assert REGISTRY.approval_reason(REGISTRY.get(name), args, tool_ctx)


def test_tests_need_approval_unless_configured(tool_ctx):
    spec = REGISTRY.get("run_tests")
    assert REGISTRY.approval_reason(spec, {}, tool_ctx)


def test_edit_requires_exactly_one_match(tool_ctx):
    call(tool_ctx, "write_file", path="f.txt", content="one two two")
    assert "appears 2 times" in call(tool_ctx, "edit_file", path="f.txt", old_text="two", new_text="2")
    assert "not found" in call(tool_ctx, "edit_file", path="f.txt", old_text="zzz", new_text="2")
    assert "Edited" in call(tool_ctx, "edit_file", path="f.txt", old_text="one", new_text="1")
    assert call(tool_ctx, "read_file", path="f.txt") == "1 two two"


def test_delete_removes_only_files(tool_ctx):
    call(tool_ctx, "write_file", path="d/f.txt", content="x")
    assert call(tool_ctx, "delete_file", path="d").startswith("ERROR")
    assert "Deleted" in call(tool_ctx, "delete_file", path="d/f.txt")


def test_blocked_command_is_refused_even_if_approved(tool_ctx):
    assert "blocked" in call(tool_ctx, "run_command", command="sudo ls")


def test_command_runs_without_a_shell(tool_ctx):
    out = call(tool_ctx, "run_command", command="echo hi && echo injected")
    assert "injected" in out and "exit code 0" in out   # echo just prints the words
    assert "hi && echo injected" in out                  # no shell, so && was not interpreted


def test_binary_file_is_refused(tool_ctx):
    (tool_ctx.workspace_dirs[0] / "b.bin").write_bytes(b"\x00\x01\x02")
    assert call(tool_ctx, "read_file", path="b.bin").startswith("ERROR")


def test_search_files(tool_ctx):
    call(tool_ctx, "write_file", path="src/a.py", content="def alpha():\n    return 1\n")
    out = call(tool_ctx, "search_files", pattern="def alpha", glob="*.py")
    assert "a.py:1" in out


def test_argument_checking(tool_ctx):
    assert "Missing required" in call(tool_ctx, "read_file")
    assert "Created" in call(tool_ctx, "write_file", path="x.txt", content="1", bogus="ignored")


def test_task_and_fact_tools(tool_ctx):
    assert "Added task #1" in call(tool_ctx, "add_task", title="buy milk", due="Friday")
    assert "buy milk" in call(tool_ctx, "list_tasks")
    assert "done" in call(tool_ctx, "complete_task", task_id=1)
    assert call(tool_ctx, "complete_task", task_id=1).startswith("ERROR")
    call(tool_ctx, "remember_fact", key="Editor", value="vim")
    assert "vim" in call(tool_ctx, "recall_facts", query="editor")
