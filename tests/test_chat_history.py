import pytest

from core.chat_history import ChatHistoryManager


def test_create_add_snapshot_and_context():
    history = ChatHistoryManager()
    history.add_message("user", "Earlier request")

    state = history.create()
    assert state.messages == []

    history.add_message("user", "Find the weather")
    history.add_message(
        "assistant",
        "",
        tool_call={"name": "weather__lookup", "parameters": {"city": "Delhi"}},
    )
    assert history.snapshot() == {
        "messages": [
            {"role": "user", "content": "Find the weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_call": {
                    "name": "weather__lookup",
                    "parameters": {"city": "Delhi"},
                },
            },
        ]
    }
    assert len(history.context(max_messages=2)) == 2


def test_delete_clears_history_and_manager_can_be_reused():
    history = ChatHistoryManager()
    history.add_message("user", "Old message")

    history.delete()
    assert history.snapshot() == {"messages": []}

    history.add_message("assistant", "New conversation")
    assert history.context() == [
        {"role": "assistant", "content": "New conversation"}
    ]


def test_history_is_bounded_and_context_limit_is_validated():
    history = ChatHistoryManager(max_messages=2)
    for message in ("one", "two", "three"):
        history.add_message("user", message)

    assert [message["content"] for message in history.context()] == ["two", "three"]
    with pytest.raises(ValueError, match="cannot be negative"):
        history.context(-1)


def test_snapshot_is_a_copy():
    history = ChatHistoryManager()
    history.add_message("user", {"request": "hello"})
    snapshot = history.snapshot()
    snapshot["messages"][0]["content"]["request"] = "changed"

    assert history.context()[0]["content"] == {"request": "hello"}


def test_rejects_invalid_message_roles_and_tool_calls():
    history = ChatHistoryManager()
    with pytest.raises(ValueError):
        history.add_message("tool", {"temperature": 28})
    with pytest.raises(ValueError):
        history.add_message("developer", "not a supported role")
    with pytest.raises(ValueError):
        history.add_message(
            "assistant", "", tool_call={"name": "", "parameters": {}}
        )


def test_resource_read_request_can_be_recorded_without_adding_resource_content():
    history = ChatHistoryManager()
    history.add_message(
        "assistant", "", resource_read={"server": "docs", "uri": "file://guide"}
    )

    assert history.context() == [{
        "role": "assistant", "content": "",
        "resource_read": {"server": "docs", "uri": "file://guide"},
    }]
