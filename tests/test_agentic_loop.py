from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from core import agentic_loop as loop_module
from core.agentic_loop import agentic_loop
from configs import mcp_settings as mcp_settings_module
from schemas.agents.tool_agent import AgentOutput


def decision(payload):
    return AgentOutput.model_validate(payload)


def test_agent_output_has_flat_schema_and_keeps_existing_fields():
    schema = AgentOutput.model_json_schema()

    assert schema["type"] == "object"
    assert "oneOf" not in schema
    assert set(schema["properties"]) == {"type", "tool_call", "resource_read", "response", "error"}
    assert decision(
        {"type": "tool_call", "tool_call": {"name": "demo__lookup", "parameters": {}}}
    ).tool_call.name == "demo__lookup"
    assert decision({"type": "respond", "response": "Done."}).response == "Done."
    assert decision({"type": "error", "error": "Unavailable."}).error == "Unavailable."
    assert decision({"type": "resource_read", "resource_read": {"server": "docs", "uri": "file://guide"}}).resource_read.uri == "file://guide"


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "tool_call"},
        {"type": "respond"},
        {"type": "error", "error": "Failed", "response": "conflict"},
        {"type": "resource_read", "resource_read": {"server": "docs", "uri": "file://guide"}, "tool_call": {"name": "demo__lookup", "parameters": {}}},
        {"type": "unknown", "response": "No"},
    ],
)
def test_agent_output_rejects_missing_or_conflicting_variant_fields(payload):
    with pytest.raises(ValueError):
        decision(payload)


@dataclass
class FakeToolResult:
    value: object

    def agent_view(self):
        return {"ok": True, "result": self.value}


class FakeMCPClient:
    def __init__(self, servers):
        self.servers = servers
        self.calls = []
        self.entered = False
        self.catalog_calls = 0

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *_):
        self.entered = False

    def tools(self):
        self.catalog_calls += 1
        return {"demo__lookup": {"description": "Look up a value"}}

    def resources(self):
        return {"docs__guide": {"name": "docs__guide", "uri": "file://guide", "description": "Project guide"}}

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return FakeToolResult({"answer": 42})

    def read_resource(self, uri, server):
        return SimpleNamespace(contents=[SimpleNamespace(uri=uri, mimeType="text/plain", text="Grounded facts")])


def test_tool_result_is_available_on_next_agent_turn(monkeypatch):
    client = FakeMCPClient([])
    monkeypatch.setattr(loop_module, "MCPClient", lambda servers: client)
    states = []
    outputs = iter(
        [
            decision(
                {
                    "type": "tool_call",
                    "tool_call": {"name": "demo__lookup", "parameters": {"id": 7}},
                }
            ),
            decision({"type": "respond", "response": "The answer is 42."}),
        ]
    )

    result = agentic_loop(
        mcp_servers=[], invoke_tool_agent_fn=lambda state: states.append(state) or next(outputs)
    ).run("Find the answer")

    assert result.status == "success"
    assert result.decision == "The answer is 42."
    assert client.calls == [("demo__lookup", {"id": 7})]
    assert states[0]["runtime"]["user_request"] == "Find the answer"
    assert states[1]["runtime"]["user_request"] == "Find the answer"
    assert states[1]["tool_results"] == [
        {"name": "demo__lookup", "ok": True, "result": {"answer": 42}}
    ]
    assert [message["role"] for message in states[0]["chat_history"]] == ["user"]
    assert [message["role"] for message in states[1]["chat_history"]] == [
        "user",
        "assistant",
    ]
    assert states[1]["chat_history"][1]["tool_call"]["name"] == "demo__lookup"
    assert states[1]["tool_results"][0]["result"] == {"answer": 42}


def test_resource_result_is_separate_and_available_on_next_turn(monkeypatch):
    client = FakeMCPClient([])
    monkeypatch.setattr(loop_module, "MCPClient", lambda servers: client)
    states = []
    outputs = iter([
        decision({"type": "resource_read", "resource_read": {"server": "docs", "uri": "file://guide"}}),
        decision({"type": "respond", "response": "Grounded answer."}),
    ])

    result = agentic_loop(mcp_servers=[], invoke_tool_agent_fn=lambda state: states.append(state) or next(outputs)).run("Use the project guide")

    assert result.status == "success"
    assert states[0]["available_resources"]["docs__guide"]["uri"] == "file://guide"
    assert states[1]["resource_results"] == [{
        "server": "docs", "uri": "file://guide",
        "contents": [{"uri": "file://guide", "mime_type": "text/plain", "text": "Grounded facts"}],
        "truncated": False,
    }]
    assert states[1]["tool_results"] == []
    assert "Grounded facts" not in str(states[1]["chat_history"])
    assert states[1]["chat_history"][1]["resource_read"] == {"server": "docs", "uri": "file://guide"}


def test_tool_catalog_is_refreshed_after_tool_execution(monkeypatch):
    class ProvisioningMCPClient(FakeMCPClient):
        def tools(self):
            self.catalog_calls += 1
            if self.catalog_calls == 1:
                return {"demo__provision": {"description": "Provision search"}}
            return {
                "demo__provision": {"description": "Provision search"},
                "demo__search": {"description": "Search"},
            }

    client = ProvisioningMCPClient([])
    monkeypatch.setattr(loop_module, "MCPClient", lambda servers: client)
    states = []
    outputs = iter([
        decision({"type": "tool_call", "tool_call": {"name": "demo__provision", "parameters": {}}}),
        decision({"type": "respond", "response": "Ready."}),
    ])

    result = agentic_loop(mcp_servers=[], invoke_tool_agent_fn=lambda state: states.append(state) or next(outputs)).run("Enable search")

    assert result.status == "success"
    assert "demo__search" not in states[0]["available_tools"]
    assert "demo__search" in states[1]["available_tools"]


def test_optional_resource_catalog_failure_is_explained_without_blocking_tools(monkeypatch):
    class NoResourcesMCPClient(FakeMCPClient):
        def resources(self):
            raise RuntimeError("server does not support resources")

    client = NoResourcesMCPClient([])
    monkeypatch.setattr(loop_module, "MCPClient", lambda servers: client)
    states = []

    result = agentic_loop(
        mcp_servers=[],
        invoke_tool_agent_fn=lambda state: states.append(state) or decision({"type": "respond", "response": "Done."}),
    ).run("Do work")

    assert result.status == "success"
    assert states[0]["available_tools"]["demo__lookup"]
    assert states[0]["available_resources"] == {}
    assert "does not support resources" in states[0]["resource_catalog_error"]


def test_agent_error_becomes_structured_failure(monkeypatch):
    client = FakeMCPClient([])
    monkeypatch.setattr(loop_module, "MCPClient", lambda servers: client)
    output = decision({"type": "error", "error": "model unavailable"})

    result = agentic_loop(
        mcp_servers=[], invoke_tool_agent_fn=lambda state: output
    ).run("Do work")

    assert result.status == "failed"
    assert result.error == "model unavailable"
    assert client.calls == []


def test_direct_response_runs_with_real_client_and_no_mcp_servers():
    result = agentic_loop(
        mcp_servers=[],
        invoke_tool_agent_fn=lambda state: decision(
            {"type": "respond", "response": "Hello."}
        ),
    ).run("Say hello")

    assert result.status == "success"
    assert result.decision == "Hello."


def test_chat_history_carries_context_across_user_turns():
    states = []
    outputs = iter(
        [
            decision({"type": "respond", "response": "First answer."}),
            decision({"type": "respond", "response": "Follow-up answer."}),
        ]
    )
    loop = agentic_loop(
        mcp_servers=[],
        invoke_tool_agent_fn=lambda state: states.append(state) or next(outputs),
    )

    first = loop.run("First question")
    second = loop.run("Follow-up question")

    assert first.status == second.status == "success"
    assert [message["content"] for message in states[1]["chat_history"]] == [
        "First question",
        "First answer.",
        "Follow-up question",
    ]


def test_missing_zapier_url_disables_remote_server(monkeypatch):
    monkeypatch.setattr(mcp_settings_module.settings, "zapier_url", None)
    monkeypatch.setattr(mcp_settings_module.settings, "zapier_token", None)

    assert mcp_settings_module.mcp_settings.servers == []


def test_tool_execution_exception_is_reported(monkeypatch):
    class FailingMCPClient(FakeMCPClient):
        def call_tool(self, name, arguments):
            raise RuntimeError("server disconnected")

    monkeypatch.setattr(loop_module, "MCPClient", lambda servers: FailingMCPClient(servers))
    call = decision(
        {"type": "tool_call", "tool_call": {"name": "demo__lookup", "parameters": {}}}
    )

    result = agentic_loop(mcp_servers=[], invoke_tool_agent_fn=lambda state: call).run("Do work")

    assert result.status == "failed"
    assert "MCP tool call failed: server disconnected" == result.error


def test_repeated_tool_calls_stop_at_iteration_limit(monkeypatch):
    client = FakeMCPClient([])
    monkeypatch.setattr(loop_module, "MCPClient", lambda servers: client)
    call = decision(
        {"type": "tool_call", "tool_call": {"name": "demo__lookup", "parameters": {}}}
    )

    result = agentic_loop(
        mcp_servers=[], invoke_tool_agent_fn=lambda state: call, max_iterations=2
    ).run("Do work")

    assert result.status == "failed"
    assert "2-iteration limit" in result.error
    assert len(client.calls) == 2


@pytest.mark.parametrize("user_input", ["", "   ", None])
def test_empty_or_invalid_request_returns_failure(user_input):
    result = agentic_loop(mcp_servers=[], invoke_tool_agent_fn=lambda state: None).run(user_input)

    assert result.status == "failed"
    assert result.error == "User request cannot be empty."


def test_max_iterations_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        agentic_loop(max_iterations=0)
