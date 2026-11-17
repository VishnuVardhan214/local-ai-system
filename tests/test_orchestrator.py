import threading

from core.llm import LLMUnavailable
from tests.conftest import collect


def types(events):
    return [e["type"] for e in events]


def text_of(events):
    return "".join(e["text"] for e in events if e["type"] == "token")


def test_coding_agent_writes_new_file_without_asking(make_env, settings):
    orch, llm, rag, approvals = make_env(replies=[
        {"tool_calls": [{"name": "write_file", "arguments": {"path": "hello.py", "content": "print(1)\n"}}]},
        "Created hello.py.",
    ])
    events = collect(orch.handle(None, "write a python script hello.py"))
    assert "approval_request" not in types(events)
    assert (settings.workspace_dirs[0] / "hello.py").read_text() == "print(1)\n"
    assert [e for e in events if e["type"] == "route"][0]["agent"] == "coding"
    assert types(events)[-1] == "done"


def test_overwrite_denied_leaves_file_alone(make_env, settings):
    target = settings.workspace_dirs[0] / "keep.py"
    target.write_text("original")
    orch, llm, rag, approvals = make_env(approval="deny", replies=[
        {"tool_calls": [{"name": "write_file", "arguments": {"path": "keep.py", "content": "changed"}}]},
        "I was not allowed to change it.",
    ])
    events = collect(orch.handle(None, "update the python file keep.py"))
    assert target.read_text() == "original"
    assert any(e["type"] == "approval_result" and e["approved"] is False for e in events)
    # the model was told it was denied
    tool_msgs = [m for m in llm.calls[-1]["messages"] if m["role"] == "tool"]
    assert "did not approve" in tool_msgs[0]["content"]


def test_overwrite_waits_for_a_real_human_decision(make_env, settings):
    target = settings.workspace_dirs[0] / "keep.py"
    target.write_text("original")
    orch, llm, rag, approvals = make_env(approval=None, replies=[
        {"tool_calls": [{"name": "write_file", "arguments": {"path": "keep.py", "content": "changed"}}]},
        "Done.",
    ])
    events = []
    worker = threading.Thread(target=lambda: events.extend(orch.handle(None, "edit python file keep.py")))
    worker.start()
    for _ in range(100):
        if approvals.pending():
            break
        threading.Event().wait(0.05)
    assert approvals.pending() and target.read_text() == "original"
    approvals.resolve(approvals.pending()[0]["id"], True)
    worker.join(5)
    assert target.read_text() == "changed"


def test_unanswered_approval_times_out_as_denied(make_env, settings):
    target = settings.workspace_dirs[0] / "keep.py"
    target.write_text("original")
    orch, llm, rag, approvals = make_env(approval=None, replies=[
        {"tool_calls": [{"name": "delete_file", "arguments": {"path": "keep.py"}}]}, "ok"])
    approvals.timeout_s = 0.2
    collect(orch.handle(None, "delete the python file keep.py"))
    assert target.exists()


def test_personal_agent_saves_task(make_env, memory):
    orch, llm, rag, approvals = make_env(replies=[
        {"tool_calls": [{"name": "add_task", "arguments": {"title": "buy milk", "due": "tomorrow"}}]},
        "Added it.",
    ])
    collect(orch.handle(None, "add a task: buy milk tomorrow"))
    assert memory.list_tasks()[0]["title"] == "buy milk"


def test_agent_cannot_use_tools_it_was_not_given(make_env, settings):
    orch, llm, rag, approvals = make_env(replies=[
        {"tool_calls": [{"name": "run_command", "arguments": {"command": "echo hi"}}]},
        "Sorry.",
    ])
    events = collect(orch.handle(None, "add a task please", agent="personal"))
    result = [e for e in events if e["type"] == "tool_result"][0]
    assert not result["ok"] and "unknown tool" in result["preview"]


def test_repeated_identical_call_is_stopped(make_env):
    call = {"tool_calls": [{"name": "list_tasks", "arguments": {}}]}
    orch, llm, rag, approvals = make_env(replies=[call, call, "Nothing new."])
    events = collect(orch.handle(None, "show my task list", agent="personal"))
    results = [e for e in events if e["type"] == "tool_result"]
    assert results[0]["ok"] and not results[1]["ok"]


def test_step_limit_forces_a_final_answer(make_env):
    loop = [{"tool_calls": [{"name": "add_note", "arguments": {"title": str(i), "text": "x"}}]} for i in range(10)]
    orch, llm, rag, approvals = make_env(replies=loop[:4] + ["final answer after limit"])
    events = collect(orch.handle(None, "note this", agent="personal"))
    assert "final answer after limit" in text_of(events)
    assert llm.calls[-1]["tools"] is None


def test_rag_agent_answers_with_sources(make_env, tmp_path):
    orch, llm, rag, approvals = make_env(replies=["Blue whales eat krill [1]."])
    f = tmp_path / "whales.txt"; f.write_text("Blue whales eat krill in huge amounts every day.")
    rag.ingest_file(f)
    events = collect(orch.handle(None, "according to the document, what do blue whales eat?"))
    src = [e for e in events if e["type"] == "sources"][0]["items"]
    assert src[0]["doc"] == "whales.txt"
    prompt = llm.calls[-1]["messages"][-1]["content"]
    assert "[1] (whales.txt" in prompt and "krill" in prompt
    assert "[1]" in text_of(events)


def test_rag_agent_without_documents_does_not_call_the_model(make_env):
    orch, llm, rag, approvals = make_env()
    events = collect(orch.handle(None, "what does the uploaded pdf say?"))
    assert "No documents are indexed" in text_of(events)
    assert llm.calls == []


def test_research_agent_seeds_passages_and_has_report_prompt(make_env, tmp_path):
    orch, llm, rag, approvals = make_env(replies=["Summary\nKrill.\nKey findings\n..."])
    f = tmp_path / "w.txt"; f.write_text("Blue whales eat krill. Krill swarms are dense near the surface.")
    rag.ingest_file(f)
    collect(orch.handle(None, "write a research report on what blue whales eat"))
    first = llm.calls[0]["messages"]
    assert "Gaps" in first[0]["content"] and "Passages already retrieved" in first[-1]["content"]
    assert {t["function"]["name"] for t in llm.calls[0]["tools"]} == {"rag_search", "list_documents"}


def test_model_failure_before_output_is_retried(make_env):
    orch, llm, rag, approvals = make_env(replies=["Hello again."])
    real = llm.stream_chat
    state = {"n": 0}

    def flaky(*a, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise LLMUnavailable("server restarting")
        return real(*a, **k)
    llm.stream_chat = flaky
    events = collect(orch.handle(None, "hello there", agent="personal"))
    assert any(e["type"] == "status" and "trying again" in e["text"] for e in events)
    assert "Hello again." in text_of(events)


def test_persistent_failure_becomes_an_error_event(make_env):
    orch, llm, rag, approvals = make_env()

    def dead(*a, **k):
        raise LLMUnavailable("down")
    llm.stream_chat = dead
    events = collect(orch.handle(None, "hello", agent="personal"))
    assert types(events)[-2:] == ["error", "done"]
    assert orch.state["status"] == "idle"


def test_context_window_trims_old_history(make_env, memory):
    orch, llm, rag, approvals = make_env(replies=["ok"])
    cid = memory.create_conversation("t")
    for i in range(30):
        memory.add_message(cid, "user" if i % 2 == 0 else "assistant", f"message {i} " + "word " * 200)
    collect(orch.handle(cid, "hello", agent="personal"))
    sent = llm.calls[0]["messages"]
    total_chars = sum(len(m["content"]) for m in sent)
    assert total_chars < (4096 - 1024) * 3 + 600
    assert sent[-1]["content"] == "hello"
    assert "message 29" in sent[-2]["content"]   # newest history survives, oldest is dropped


def test_old_history_is_summarized_after_reply(make_env, memory):
    orch, llm, rag, approvals = make_env(replies=["answer", "Summary of the chat so far."])
    cid = memory.create_conversation("t")
    for i in range(14):
        memory.add_message(cid, "user" if i % 2 == 0 else "assistant", "text " * 150)
    collect(orch.handle(cid, "one more thing", agent="personal"))
    summary, remaining = memory.prompt_history(cid)
    assert summary == "Summary of the chat so far." and len(remaining) <= 8


def test_only_one_request_runs_at_a_time(make_env):
    orch, llm, rag, approvals = make_env(replies=["a", "b"])
    gen1 = orch.handle(None, "hi", agent="personal")
    next(gen1); next(gen1)          # session event, then it acquires the gate on the second step
    assert orch.gate.locked()
    gen1.close()
    assert not orch.gate.locked()
