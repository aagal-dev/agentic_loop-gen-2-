from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        description="Exact name of an exposed MCP tool."
    )
    parameters: dict[str, Any] = Field(
        description="Tool arguments as a JSON object. Use {} if no arguments are required."
    )


class ResourceReadRequest(BaseModel):
    """Identify an MCP resource to read from its owning server."""

    model_config = ConfigDict(extra="forbid")

    server: str = Field(min_length=1, description="Exact MCP server name from available_resources.")
    uri: str = Field(min_length=1, description="Exact resource URI from available_resources.")


class AgentOutput(BaseModel):
    """One tool call, resource read, direct response, or structured error."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_call", "resource_read", "respond", "error"]
    tool_call: ToolInvocation | None = None
    resource_read: ResourceReadRequest | None = None
    response: str | None = Field(default=None, min_length=1)
    error: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_output_fields(self) -> "AgentOutput":
        active_field = {
            "tool_call": self.tool_call,
            "resource_read": self.resource_read,
            "respond": self.response,
            "error": self.error,
        }[self.type]
        if active_field is None:
            raise ValueError(f"'{self.type}' output requires its matching field.")

        inactive_fields = {
            "tool_call": self.tool_call if self.type != "tool_call" else None,
            "resource_read": self.resource_read if self.type != "resource_read" else None,
            "response": self.response if self.type != "respond" else None,
            "error": self.error if self.type != "error" else None,
        }
        present_inactive_fields = [name for name, value in inactive_fields.items() if value is not None]
        if present_inactive_fields:
            raise ValueError(
                "Only the field matching 'type' may be provided; conflicting fields: "
                + ", ".join(present_inactive_fields)
            )
        return self
