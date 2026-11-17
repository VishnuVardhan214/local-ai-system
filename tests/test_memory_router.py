from agents import Agent, build_agents, register_agent
from core.mock import MockLLM
from orchestrator.router import Router


def test_conversation_roundtrip_and_delete(memory):
    cid = memory.create_conversation("hello")
    memory.add_message(cid, "user", "hi")
    memory.add_message(cid, "assistant", "hey", agent="personal", meta={"tools": []})
    msgs = memory.get_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["agent"] == "personal"
    memory.delete_conversation(cid)
    assert not memory.conversation_exists(cid)


def test_old_messages_fold_into_summary(memory):
    cid = memory.create_conversation("t")
    ids = [memory.add_message(cid, "user" if i % 2 == 0 else "assistant", "x" * 500) for i in range(20)]
    old = memory.messages_to_fold(cid, keep_last=6, threshold_chars=3000)
    assert len(old) == 14
    memory.set_summary(cid, "short summary", old[-1]["id"])
    summary, rest = memory.prompt_history(cid)
    assert summary == "short summary" and len(rest) == 6


def test_short_conversations_are_not_summarized(memory):
    cid = memory.create_conversation("t")
    for i in range(4):
        memory.add_message(cid, "user", "hi")
    assert memory.messages_to_fold(cid, 6, 3000) == []


def test_facts_overwrite_and_forget(memory):
    memory.remember("City", "Pune"); memory.remember("city", "Mumbai")
    assert memory.recall("city") == [{"key": "city", "value": "Mumbai"}]
    assert memory.forget("city") and memory.recall("city") == []


def test_router_keywords():
    r = Router(build_agents(), MockLLM())
    assert r.route("fix the bug in utils.py")[0] == "coding"
    assert r.route("add a task to buy milk tomorrow")[0] == "personal"
    assert r.route("what does the uploaded pdf say about revenue?")[0] == "rag"
    assert r.route("write a report comparing the two papers")[0] == "research"


def test_router_manual_choice_and_unknown(memory):
    r = Router(build_agents(), MockLLM())
    assert r.route("anything", "research") == ("research", "chosen by you")
    assert r.route("anything", "nope")[0] == "personal"


def test_router_falls_back_to_model_then_default():
    llm = MockLLM(replies=["coding"])
    r = Router(build_agents(), llm)
    assert r.route("hello there general kenobi") == ("coding", "chosen by the model")
    llm2 = MockLLM(replies=["gibberish"])
    assert Router(build_agents(), llm2).route("hello there")[0] == "personal"


def test_new_agent_joins_routing_automatically():
    from agents import AGENT_CLASSES

    @register_agent
    class WeatherAgent(Agent):
        name = "weather_test"
        description = "weather"
        keywords = ("forecast",)
    try:
        r = Router(build_agents(), MockLLM())
        assert r.route("what is the forecast")[0] == "weather_test"
    finally:
        AGENT_CLASSES.pop("weather_test", None)
