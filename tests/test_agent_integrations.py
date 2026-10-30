from pydantic import BaseModel

from integrations.base_agent import BaseAgent
from integrations.ollama_client import OllamaClient
from schemas.agents.tool_agent import AgentOutput


class FailedClient:
    def generate(self, **kwargs):
        from integrations.ollama_client import LLMResponse

        return LLMResponse[AgentOutput](success=False, error="offline")


def test_base_agent_converts_provider_errors_to_agent_error_output():
    agent = BaseAgent("prompt", AgentOutput, client=FailedClient())

    result = agent.invoke({"runtime": {"user_request": "hello"}})

    assert result.type == "error"
    assert result.error == "offline"


def test_resource_read_output_is_accepted_by_agent_schema():
    result = AgentOutput.model_validate({
        "type": "resource_read",
        "resource_read": {"server": "docs", "uri": "file://guide"},
    })
    assert result.resource_read.server == "docs"


def test_ollama_client_validates_structured_response():
    class ResponseModel(BaseModel):
        answer: int

    class FakeOllama:
        def chat(self, **kwargs):
            assert kwargs["messages"][1]["content"] == "{'question': 'life'}"
            return {"message": {"content": '{"answer": 42}'} }

    client = OllamaClient.__new__(OllamaClient)
    client.model = "test-model"
    client.max_token_output = 100
    client.temperature = 0
    client.client = FakeOllama()

    response = client.generate({"question": "life"}, ResponseModel, "system")

    assert response.success
    assert response.data == ResponseModel(answer=42)


def test_ollama_client_retries_then_returns_controlled_error_for_invalid_json():
    class ResponseModel(BaseModel):
        answer: int

    class FakeOllama:
        calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            return {"message": {"content": "not json"}}

    ollama_client = FakeOllama()
    client = OllamaClient.__new__(OllamaClient)
    client.model = "test-model"
    client.max_token_output = 100
    client.temperature = 0
    client.client = ollama_client

    response = client.generate({}, ResponseModel)

    assert not response.success
    assert ollama_client.calls == 3
    assert "after 3 attempts" in response.error
    assert "validation error" not in response.error.lower()


def test_ollama_client_retries_non_json_prose_then_accepts_valid_json():
    class FakeOllama:
        calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"message": {"content": "Based on my web searches, here is the answer."}}
            assert "required JSON schema" in kwargs["messages"][-1]["content"]
            assert kwargs["format"] == AgentOutput.model_json_schema()
            return {
                "message": {
                    "content": '{"type":"respond","response":"Here is the answer."}'
                }
            }

    ollama_client = FakeOllama()
    client = OllamaClient.__new__(OllamaClient)
    client.model = "test-model"
    client.max_token_output = 100
    client.temperature = 0
    client.client = ollama_client

    response = client.generate({"query": "research"}, AgentOutput, "Use JSON.")

    assert response.success
    assert ollama_client.calls == 2
    assert response.data.response == "Here is the answer."


def test_persistent_prose_becomes_agent_error_instead_of_validation_exception():
    class FakeOllama:
        def chat(self, **kwargs):
            return {"message": {"content": "Based on my web searches, here is the answer."}}

    client = OllamaClient.__new__(OllamaClient)
    client.model = "test-model"
    client.max_token_output = 100
    client.temperature = 0
    client.client = FakeOllama()
    agent = BaseAgent("prompt", AgentOutput, client=client)

    output = agent.invoke({"runtime": {"user_request": "research"}})

    assert output.type == "error"
    assert "after 3 attempts" in output.error


def test_ollama_client_escapes_raw_newlines_in_json_strings():
    class FakeOllama:
        def chat(self, **kwargs):
            return {
                "message": {
                    "content": '{"type":"respond","response":"First line\nSecond line"}'
                }
            }

    client = OllamaClient.__new__(OllamaClient)
    client.model = "test-model"
    client.max_token_output = 100
    client.temperature = 0
    client.client = FakeOllama()

    response = client.generate({}, AgentOutput)

    assert response.success
    assert response.data.response == "First line\nSecond line"


def test_ollama_client_escapes_raw_newlines_in_tool_call_arguments():
    class FakeOllama:
        def chat(self, **kwargs):
            return {
                "message": {
                    "content": (
                        '{"type":"tool_call","tool_call":{"name":"demo__search",'
                        '"parameters":{"query":"first line\nsecond line"}}}'
                    )
                }
            }

    client = OllamaClient.__new__(OllamaClient)
    client.model = "test-model"
    client.max_token_output = 100
    client.temperature = 0
    client.client = FakeOllama()

    response = client.generate({}, AgentOutput)

    assert response.success
    assert response.data.tool_call.parameters["query"] == "first line\nsecond line"


def test_ollama_client_does_not_repair_control_characters_outside_strings():
    class FakeOllama:
        def chat(self, **kwargs):
            return {"message": {"content": '{\n"type":"respond",\u0000"response":"Hi"}'}}

    client = OllamaClient.__new__(OllamaClient)
    client.model = "test-model"
    client.max_token_output = 100
    client.temperature = 0
    client.client = FakeOllama()

    response = client.generate({}, AgentOutput)

    assert not response.success
