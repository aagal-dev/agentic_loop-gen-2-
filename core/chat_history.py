"""A bounded, JSON-ready conversation history for agent context."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from schemas.agents.tool_agent import ResourceReadRequest, ToolInvocation

ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """One conversation entry, including optional tool-call metadata."""

    model_config = ConfigDict(extra="forbid")

    role: ChatRole
    content: JsonValue
    tool_call: ToolInvocation | None = None
    resource_read: ResourceReadRequest | None = None


class ChatHistoryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(default_factory=list)


class ChatHistoryManager:
    """Store recent messages and provide snapshots or a compact agent view."""

    def __init__(self, max_messages: int = 40) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        self.max_messages = max_messages
        self._state = ChatHistoryState()

    def create(self) -> ChatHistoryState:
        """Start a fresh history and return its empty state."""
        self._state = ChatHistoryState()
        return self._state.model_copy(deep=True)

    def delete(self) -> None:
        """Clear the conversation while keeping the manager reusable."""
        self._state.messages.clear()

    def add_message(
        self,
        role: ChatRole,
        content: JsonValue,
        *,
        tool_call: ToolInvocation | dict[str, Any] | None = None,
        resource_read: ResourceReadRequest | dict[str, str] | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            role=role,
            content=content,
            tool_call=tool_call,
            resource_read=resource_read,
        )
        self._state.messages.append(message)
        if len(self._state.messages) > self.max_messages:
            del self._state.messages[: len(self._state.messages) - self.max_messages]
        return message.model_copy(deep=True)

    def snapshot(self) -> dict[str, Any]:
        """Return an independent JSON-ready copy of the stored history."""
        return self._state.model_dump(mode="json", exclude_none=True)

    def context(self, max_messages: int = 20) -> list[dict[str, Any]]:
        """Return the most recent messages in the shape intended for agents."""
        if max_messages < 0:
            raise ValueError("max_messages cannot be negative")
        selected = self._state.messages[-max_messages:] if max_messages else []
        return [message.model_dump(mode="json", exclude_none=True) for message in selected]
