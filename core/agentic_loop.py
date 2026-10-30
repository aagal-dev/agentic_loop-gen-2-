from typing import Any, Callable

from agents.tool_agent import invoke_tool_agent
from configs.mcp_settings import mcp_settings
from core.chat_history import ChatHistoryManager
from core.runtime_state import RuntimeStateManager
from packages.MCP import MCPClient
from packages.MCP.client.result import resource_agent_view
from schemas.agents.tool_agent import AgentOutput
from schemas.system.agent_loop_contract import AgentLoopResult


class agentic_loop:
    """Run an agent decision loop and execute its MCP tool requests."""

    def __init__(
        self,
        mcp_servers=None,
        invoke_tool_agent_fn: Callable | None = None,
        max_iterations: int = 20,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.mcp_servers = mcp_settings.servers if mcp_servers is None else mcp_servers
        self.invoke_tool_agent = invoke_tool_agent_fn or invoke_tool_agent
        self.max_iterations = max_iterations
        self.runtime = RuntimeStateManager()
        self.chat_history = ChatHistoryManager()

    def run(self, user: str) -> AgentLoopResult:
        if not isinstance(user, str) or not user.strip():
            return AgentLoopResult(
                status="failed", error="User request cannot be empty."
            )

        self.runtime.create(user_request=user.strip(), steps=[])
        self.chat_history.add_message("user", user.strip())
        tool_results = []
        resource_results = []

        try:
            with MCPClient(self.mcp_servers) as mcp:
                available_tools = mcp.tools()
                available_resources, resource_catalog_error = self._resource_catalog(mcp)
                for _ in range(self.max_iterations):
                    agent_state = {
                        "available_tools": available_tools,
                        "available_resources": available_resources,
                        "resource_catalog_error": resource_catalog_error,
                        "runtime": self.runtime.agent_context(),
                        "chat_history": self.chat_history.context(),
                        "tool_results": list(tool_results),
                        "resource_results": list(resource_results),
                    }
                    decision = self.invoke_tool_agent(state=agent_state)
                    if not isinstance(decision, AgentOutput):
                        return self._failure("Tool agent returned an invalid decision.")

                    output = decision
                    if output.type == "error":
                        self.chat_history.add_message("assistant", output.error)
                        return self._failure(output.error)
                    if output.type == "respond":
                        self.chat_history.add_message("assistant", output.response)
                        return AgentLoopResult(
                            status="success",
                            decision=output.response,
                            state=self.runtime.snapshot(),
                        )

                    if output.type == "resource_read":
                        if output.resource_read is None:
                            return self._failure("Resource read decision did not include resource_read data.")
                        request = output.resource_read
                        try:
                            self.chat_history.add_message("assistant", "", resource_read=request)
                            raw_result = mcp.read_resource(request.uri, request.server)
                            resource_results.append({
                                "server": request.server,
                                "uri": request.uri,
                                **resource_agent_view(raw_result),
                            })
                            available_tools = mcp.tools()
                            available_resources, resource_catalog_error = self._resource_catalog(mcp)
                        except Exception as exc:
                            return self._failure(f"MCP resource read failed: {exc}")
                        continue

                    if output.tool_call is None:
                        return self._failure("Tool call decision did not include tool_call data.")

                    try:
                        print(f"\nTOOL CALL: name > {output.tool_call.name} | params > {output.tool_call.parameters}")
                        self.chat_history.add_message(
                            "assistant",
                            "",
                            tool_call=output.tool_call,
                        )
                        result = mcp.call_tool(
                            output.tool_call.name, output.tool_call.parameters
                        )
                    except Exception as exc:
                        return self._failure(f"MCP tool call failed: {exc}")

                    result_view = result.agent_view()
                    tool_results.append(
                        {
                            "name": output.tool_call.name,
                            **result_view,
                        }
                    )
                    available_tools = mcp.tools()
                    available_resources, resource_catalog_error = self._resource_catalog(mcp)

            return self._failure(
                f"Agent exceeded the {self.max_iterations}-iteration limit."
            )
        except Exception as exc:
            return self._failure(f"Agent loop failed: {exc}")

    def _failure(self, error: str) -> AgentLoopResult:
        return AgentLoopResult(
            status="failed",
            decision=None,
            state=self.runtime.snapshot(),
            error=error,
        )

    @staticmethod
    def _resource_catalog(mcp) -> tuple[dict[str, Any], str | None]:
        """Keep optional resource support from breaking otherwise valid tool use."""
        try:
            return mcp.resources(), None
        except Exception as exc:
            return {}, f"MCP resource catalog unavailable: {exc}"
