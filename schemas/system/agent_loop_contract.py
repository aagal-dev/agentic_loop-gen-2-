from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Literal


class AgentLoopResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "failed", "partial"]

    decision: Any | None = None

    error: str | None = None

    state: dict[str, Any] = Field(default_factory=dict)